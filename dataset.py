import numpy as np
from sklearn import preprocessing
import torch

RANDOM_SEED: int = 3948576

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: np.ndarray) -> None:

        # Set random seed        
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.rng.shuffle(dataset)

        # Extract data and labels
        self._data = dataset[:, :-1]
        self.labels = dataset[:,-1].reshape(-1)

        # Select element sets
        self.data = {
            'mee': self._data[:, :6],
            'pnc': self._data[:, 6:12],
            'qtn': self._data[:, 12:]
        }

        # Slice into training, validation, test sets
        n = len(self._data)
        train_idx = int(0.7 * n)
        valid_idx = int(0.8 * n)

        self.train = {}
        self.valid = {}
        self.test = {}

        self.train_labels, self.valid_labels, self.test_labels = np.split(self.labels, [train_idx, valid_idx])

        elements = ['mee', 'pnc', 'qtn']
        for el in elements:
            (
                self.train[el],
                self.valid[el],
                self.test[el]
            ) = np.split(self.data[el], [train_idx, valid_idx])
        
        # Scale data from training data
        # TODO Add minmax scaler for altitude measures
        for el in elements:
            scaler = preprocessing.StandardScaler().fit(self.train[el])
            self.train[el] = scaler.transform(self.train[el])

        self.subset = None
        self.sublab = None

        return
    
    def __len__(self) -> int:
        return len(self.subset)
    
    def __getitem__(self, idx):
        return self.subset[idx], self.sublab[idx]

    def tensorify(self, data_dict: dict, data_type:torch.dtype=torch.float32) -> dict:
        """
        Input dictionary of np.ndarrays
        Ouput dictionary of torch.tensor
        """

        for key in data_dict.keys():
            data_dict[key] = torch.tensor(data_dict[key]).to(data_type)

        return data_dict
    
    def choose_set(self, settype:str, element:str) -> None:

        if settype == 'train':
            dataset = self.tensorify(self.train, torch.float32)
            labels = torch.tensor(self.train_labels).to(torch.int64)
        elif settype == 'valid':
            dataset = self.tensorify(self.valid, torch.float32)
            labels = torch.tensor(self.valid_labels).to(torch.int64)
        elif settype == 'test':
            dataset = self.tensorify(self.test, torch.float32)
            labels = torch.tensor(self.test_labels).to(torch.int64)
        else:
            return

        if element not in ['mee', 'pnc', 'qtn']:
            return
        
        self.subset = dataset[element]
        self.sublab = labels

        return
