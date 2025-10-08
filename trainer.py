# Import PyTorch
import torch

# Data libraries
import dataset
import manip
import preprocessing

# Progress visual
from tqdm import tqdm

# Get compute device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define model architecture function
def create_model():
    return torch.nn.Sequential(
        torch.nn.Linear(6, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, 2),
        torch.nn.ReLU(),
        torch.nn.LogSoftmax(dim=1)
    ).to(device=device, non_blocking=True)

# Create separate model instances
num_models = 30
models = [create_model() for _ in range(num_models)]

# Create compute streams
streams = [torch.cuda.Stream() for _ in range(num_models)]

print(f"Preparing Data")
# Get pair data
path = './data/'
data, basenames = manip.loader(path)
data, basenames = manip.drop_nan(data, basenames)
pairs = manip.generate_pairs(data, basenames)
pairs = manip.shuffle_balance_data(pairs)

# Preprocess
pair_data = pairs[:, :-1]
pair_data = preprocessing.process_data(pair_data)
pair_labels = pairs[:, -1]

# Create separate datasets for each model
datasets = []
batch_size = 3000
for _ in range(num_models):
    ds = dataset.PairDataset()
    ds.load_dataset(pair_data)
    ds.load_labels(pair_labels)
    ds.choose_subset('train')
    datasets.append(
        torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
    )

# Create separate dataloaders
iters = [iter(ds) for ds in datasets]

# Create optimizers
optimizers = [
    torch.optim.AdamW(mdl.parameters(), lr=0.002) for mdl in models
]

# Loss criterion
criterion = torch.nn.CrossEntropyLoss()

# Training function
def get_gpu_batch(model_idx):
    """Slices the global GPU data tensor using shuffled indices."""
    
    current_start_idx = model_current_batch_indices[model_idx]
    
    # Check for end of epoch
    if current_start_idx >= total_samples:
        return None, None, True # Signal epoch end

    # Get the current set of indices for this batch
    indices = model_epoch_indices[model_idx]
    
    end_idx = min(current_start_idx + batch_size, total_samples)
    batch_indices = indices[current_start_idx:end_idx]
    
    # Update the next starting point
    model_current_batch_indices[model_idx] = end_idx
    
    # Slice the global GPU tensors
    inputs = gpu_pair_data[batch_indices]
    labels = gpu_pair_labels[batch_indices]
    
    return inputs, labels, False # Success

print(f"Training...")
# Main training loop
num_epochs = 5
epoch = 0
completed = [False] * num_models

with tqdm(total=num_epochs * num_models, unit=f"epoch(s)") as pbar:

    while epoch < num_epochs:

        if all(completed):
            epoch += 1
            completed = [False] * num_models

        for i in range(num_models):
            if completed[i]:
                continue

            with torch.cuda.stream(streams[i]):
                
                model = models[i]
                optim = optimizers[i]

                try:
                    inputs, labels = next(iters[i])
                    
                    optim.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optim.step()
                    
                except StopIteration:
                    completed[i] = True
                    iters[i] = iter(datasets[i])
                    pbar.update(1)
                    continue

