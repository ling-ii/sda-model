import os
import tempfile
import ray.cloudpickle as pickle
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.tune import Checkpoint
from ray.tune.search.optuna import OptunaSearch
from functools import partial

import dataset  # Your dataset module

# Enable CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TunableModel(nn.Module):
    def __init__(self, config):
        super(TunableModel, self).__init__()
        
        # Extract architecture params from config
        input_size = 6  # Assuming 6 features as input
        output_size = 2  # Assuming 2 classes
        layer_width = config["layer_width"]
        num_layers = config["num_layers"]
        
        # Build dynamic architecture
        layers = []
        
        # First layer: input to hidden
        layers.append(nn.Linear(input_size, layer_width))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(layer_width, layer_width))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(layer_width, output_size))
        
        # Build sequential model
        self.model = nn.Sequential(*layers)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.model(x)
        return self.log_softmax(x)

def load_data(data_path='./data/dataset.npy'):
    """Load combined features and labels from binary file"""
    print(f"Loading binary dataset from {data_path}...")
    
    # Load numpy file
    combined_data = np.load(data_path)
    
    # Reshape the flat array - assuming 6 features + 1 label = 7 columns
    # If your dimensions are different, adjust accordingly
    num_samples = combined_data.size // 7
    combined_data = combined_data.reshape(num_samples, 7)
    
    # Split into features and labels
    features = combined_data[:, :-1]  # All columns except the last
    labels = combined_data[:, -1]     # Last column
    
    print(f"Loaded data with shape: {features.shape}, labels shape: {labels.shape}")
    
    # Create dataset
    ds = dataset.PairDataset()
    ds.load_dataset(features)
    ds.load_labels(labels)
    
    return ds

def train_model(config, data=None, checkpoint_dir=None):
    """Training function for Ray Tune"""
    # Load model
    model = TunableModel(config).to(device)
    
    # Set up loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config["lr"],
    )
    
    # Load checkpoint if provided
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint.pt")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    # Create new dataset instances for train and validation
    # Instead of using data.copy() which doesn't exist
    train_data = dataset.PairDataset()
    train_data.load_dataset(data.data)  # Use the original data array
    train_data.load_labels(data.labels) # Use the original labels array
    train_data.choose_subset('train')   # Use built-in split functionality
    
    val_data = dataset.PairDataset()
    val_data.load_dataset(data.data)
    val_data.load_labels(data.labels)
    val_data.choose_subset('valid')     # 'valid' is used in PairDataset, not 'val'
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Training loop
    for epoch in range(10):  # Fixed number of epochs for tuning
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            # Note: Data should already be on GPU from dataset.choose_subset()
            # But just in case, ensure they're on the right device
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        # Create checkpoint directory and save model
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            
            # Save model and optimizer states
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict()
                },
                checkpoint_path
            )
            
            # Create checkpoint from directory
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            
            # Report metrics to Ray Tune with proper checkpoint
            train.report({
                "loss": avg_val_loss,
                "accuracy": val_accuracy,
                "epoch": epoch
            }, checkpoint=checkpoint)

def tune_model():
    """Set up and run hyperparameter tuning"""
    # Load data once to be shared across trials
    data = load_data()
    
    # Define search space
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([128, 256, 512, 1024, 2048]),
        "layer_width": tune.choice([64, 128, 200, 256, 512]),
        "num_layers": tune.choice([2, 3, 4, 5, 6])
    }
    
    # Set up scheduler
    scheduler = ASHAScheduler(
        max_t=10,
        grace_period=2,
        reduction_factor=2
    )
    
    # Set up search algorithm
    search_alg = OptunaSearch()
    
    # Run tuning
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model, data=data),
            resources={"cpu": 2, "gpu": 0.2}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=50,
            max_concurrent_trials=5
        ),
        param_space=config,
        run_config=ray.air.RunConfig(
            name="sda_model_tuning",
            # Convert relative path to absolute path with file:// prefix
            storage_path=f"file://{os.path.abspath('./ray_results')}",
            verbose=1
        )
    )
    
    results = tuner.fit()
    
    # Get best trial
    best_trial = results.get_best_result(metric="loss", mode="min")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.metrics['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.metrics['accuracy']}")
    
    # Save best model config
    with open("best_model_config.txt", "w") as f:
        for key, val in best_trial.config.items():
            f.write(f"{key}: {val}\n")
    
    return best_trial

def train_best_model(best_config):
    """Train model with best hyperparameters for full duration"""
    # Load data
    data = load_data()
    data.choose_subset('train')  # Use full training set
    
    # Create model with best config
    model = TunableModel(best_config).to(device)
    
    # Create DataLoader
    train_loader = DataLoader(
        dataset=data,
        batch_size=int(best_config["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=best_config["lr"])
    num_epochs = 500  # Full training duration
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader)}")
    
    # Save final model
    torch.save(model.state_dict(), "best_model.pt")
    return model

if __name__ == "__main__":
    # Initialize Ray
    ray.init(num_cpus=4, num_gpus=1)
    
    # Run hyperparameter tuning
    best_trial = tune_model()
    
    # Train best model
    best_model = train_best_model(best_trial.config)
    
    print("Training complete!")