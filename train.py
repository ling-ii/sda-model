import dataset
import manip
import model
import preprocessing
import multitrainer

# import multiprocessing as mp
from tqdm import tqdm
import torch
import os
import re
import pandas as pd

def create_layers(neurons: list[int], nonlin=torch.nn.ReLU) -> list:
    
    n_layers = len(neurons)
    if n_layers < 3:
        raise UserWarning(f"Number of layers < 3. Cannot create layers.")

    # Construct layers
    layers = []
    for i in range(n_layers-1):
        layers.append(torch.nn.Linear(neurons[i], neurons[i+1]))
        layers.append(nonlin())

    #  Add log soft max to classify
    layers.append(torch.nn.LogSoftmax(dim=1))
    
    return layers

# def train_model(args) -> tuple:
#     mdl, mdl_id = args
#     try:
#         result = mdl.train()
#         return mdl_id, result
#     except Exception as e:
#         return mdl_id, None
    
def write_report(results:dict, path:str='./training/', *params: list) -> None:

    # Get latest batch id
    batches = [
        d for d in os.listdir(path)
        if os.path.isdir(d) and re.search('^batch_[0-9]+$', d)
    ]
    batch_ids = [int(re.search('[0-9]+', batch).group()) for batch in batches]
    batch_id = str(max(batch_ids) + 1) if batch_ids else '0'
    batch_name = os.path.join(path, f"batch_{batch_id}")

    # Create batch directory
    os.makedirs(batch_name, exist_ok=True)

    # Write results
    for mdl_id, result in results.items():
        if result is not None:
            mdl_name = os.path.join(batch_name, f'model_{mdl_id}.np')
            result.tofile(mdl_name)
            print(f"Model {mdl_id} saved to {mdl_name}")
        else:
            print(f"Model {mdl_id} training failed")

    mdl_report_name = os.path.join(batch_name, f"report_{batch_id}")
    param_df = pd.DataFrame(params).T
    param_df.to_csv(mdl_report_name,index=False)

    return

def main() -> None:

    # Data load, cleaning, and generate pairs
    path = './data/'
    print(f"Loading dataset from '{path}' ...")
    data, basenames = manip.loader(path)
    data, basenames = manip.drop_nan(data, basenames)
    pairs = manip.generate_pairs(data, basenames)
    pairs = manip.shuffle_balance_data(pairs)

    # Preprocess
    print(f"Preprocessing data ...")
    pair_data = pairs[:, :-1]
    pair_data = preprocessing.process_data(pair_data)
    pair_labels = pairs[:, -1]

    # Load data into model dataloader
    print(f"Loading data ...")
    ds = dataset.PairDataset()
    ds.load_dataset(pair_data)
    ds.load_labels(pair_labels)
    ds.choose_subset('train')

    # Create list of model parameters
    print(f"Loading model parameters ...")
    neurons = [
        # Vary batch size
        [6, 200, 200, 200, 2],
        [6, 200, 200, 200, 2],
        [6, 200, 200, 200, 2],
        [6, 200, 200, 200, 2],
        [6, 200, 200, 200, 2],
        
        # Vary learning rate
        [6, 200, 200, 200, 2],
        [6, 200, 200, 200, 2],
        [6, 200, 200, 200, 2],
        [6, 200, 200, 200, 2],
        [6, 200, 200, 200, 2],
        
        # Vary layers
        [6, 200, 200, 200, 2],
        [6, 400, 400, 400, 2],
        [6, 200, 200, 200, 200, 200, 2],
        [6, 400, 400, 400, 400, 400, 2],
        [6, 500, 500, 2],
    ]
    layers = [create_layers(layer) for layer in neurons]

    lrs = [
        0.002,
        0.002,
        0.002,
        0.002,
        0.002,

        0.003,
        0.004,
        0.005,
        0.006,
        0.007,

        0.002,
        0.002,
        0.002,
        0.002,
        0.002,
    ]

    batch_sizes = [
        2750,
        2875, 
        3000, 
        3125,
        3250,

        3000,
        3000,
        3000,
        3000,
        3000,

        3000,
        3000,
        3000,
        3000,
        3000,
    ]

    num_epochs = [
        500,
        500,
        500,       
        500,
        500,

        500,
        500,
        500,
        500,
        500,
        
        500,
        500,
        500,
        500,
        500,
    ]

    # Check parameters
    params = [layers, lrs, batch_sizes, num_epochs]
    if len(set([len(param) for param in params])) != 1:
        raise UserWarning(f"Parameter lists are not all of equal length.")

    # Create list of models with parameters from lists, keep the same dataset
    print(f"Creating model objects ...")
    num_mdls: int = len(layers)
    mdls: list[model.Model] = []
    for i in range(num_mdls):

        # Instantiate model
        mdl = model.Model(
            layers[i],
            ds, 
            lrs[i], 
            batch_sizes[i], 
            num_epoch=num_epochs[i],
            mdl_id=i
        )

        # Add model to list
        mdls.append(mdl)

    # Run training on all models and create reports
    print(f"Running model training ...")

    # Create tuple list of models and ids
    # mdl_args = [(mdl, i) for i, mdl in enumerate(mdls)]

    # Run multiprocessing pool to train models in parallel
    # with mp.Pool(processes=min(num_mdls, mp.cpu_count())) as pool:
    #     results = list(
    #         tqdm(
    #             pool.imap(train_model, mdl_args),
    #             total=num_mdls,
    #             desc="Training models",
    #             unit="model"
    #         )
    #     )

    trainer = multitrainer.MultiTrainer(mdls)
    results = trainer.train_models()

    # Process results
    print(f"Training complete. Processing results ...")
    write_report(results, 'training/', neurons, lrs, batch_sizes, num_epochs)

if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    main()