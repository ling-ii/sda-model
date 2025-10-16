import torch
import yaml
import os
import re

def read_model_config(path: str) -> dict:

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} model config not found")
    
    with open(path, 'r') as fp:
        config = yaml.safe_load(fp)

    return config

def construct_model(config: dict):

    try:
        layer_width = config['layer_width']
        num_layers = config['num_layers']
    except KeyError:
        raise KeyError(f"Config file does not contain correct keys")

    layers = []

    layers.append(torch.nn.Linear(6, layer_width))
    layers.append(torch.nn.ReLU())

    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(layer_width, layer_width))
        layers.append(torch.nn.ReLU())

    layers.append(torch.nn.Linear(layer_width, 2))
    layers.append(torch.nn.LogSoftmax(dim=1))

    return torch.nn.Sequential(*layers)

def clean_state_keys(state_dict: dict):
    clean_state_dict = {}
    for k, v in state_dict.items():
        if s := re.search('^(.+\.)?(\d+\.)(.+)$', k):
            k_new = ''.join(s.groups()[1:])
            clean_state_dict[k_new] = v
        else:
            raise KeyError(f"Unexpected key in state dict {k}")

    return clean_state_dict

def load_model_weights(
    model: torch.nn.Sequential,
    path: str
) -> torch.nn.Sequential:
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} model weight dict not found")

    state_dict = clean_state_keys(torch.load(path))
    model.load_state_dict(state_dict)
    
    return model

def main():
    config = read_model_config('best_model_config.txt')
    model  = construct_model(config)
    model  = load_model_weights(model, 'best_model.pt')

if __name__ =='__main__':
    main()
