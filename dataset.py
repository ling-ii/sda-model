import numpy as np
import numpy.typing as npt
import torch

__RANDOM_SEED__: int = 3948576

class PairDataset(torch.utils.data.Dataset):

    """
        # PairDataset

        Create dataset object with data and label loading functions.
        
        ## Data structure:

        +---------+---------+---------+       +--------+
        | MEE * 3 | PNC * 3 | QTN * 3 |   +   | labels |
        +---------+---------+---------+       +--------+

        - MEE: (h, k) -> proper ecc., (p, q) -> proper inc., proper semi-major
        - PNC: similar to MEE but with different scalings
        - QTN: (qtn0, qtn3), (qtn1, qtn2), proper ecc.

    """

    def __init__(self) -> None:

        # Complete data
        self.rng: np.Generator = np.random.default_rng(__RANDOM_SEED__)
        self.data: npt.ArrayLike = None
        self.labels: npt.ArrayLike = None

        # Data splitting ratios
        self._train_split: int = 0.7
        self._valid_split: int = 0.8

        # Data subsets and labels
        self._subsets: tuple[npt.ArrayLike] = None
        self._sublabs: tuple[npt.ArrayLike] = None
        self.subset: torch.Tensor = None
        self.sublab: torch.Tensor = None

        return

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple:
        return self.subset[idx], self.sublab[idx]
    
    def __totensor__(self, data: npt.ArrayLike) -> torch.Tensor:
        return torch.tensor(data)

    def __checkset__(self) -> bool:
        return (self.data.shape[0] == self.labels.shape[0])

    def __splitset__(self, set) -> None:
        n = len(set)
        return np.split(set, [int(self._train_split *n), int(self._valid_split)])

    def load_dataset(self, dataset: npt.ArrayLike) -> None:
        self.data = np.asarray(dataset, dtype=np.float32).copy()
        return

    def load_labels(self, labels: npt.ArrayLike) -> None:
        self.labels = np.asarray(labels, dtype=np.int32).copy()
        self.labels.reshape(-1, 1)
        return
    
    def set_train_split(self, split: float) -> None:
        self._train_split = split
        return

    def set_valid_split(self, split: float) -> None:
        self.valid_split = split
        return
    
    def choose_subset(self, subset: str) -> None:
        if subset not in ('train', 'valid', 'test'):
            raise ValueError(f"Invalid subset type: {subset}")

        if (self.data is None) or (self.labels is None):
            raise AssertionError(f"Dataset or labels have not been loaded.")

        if (self._subsets is None) or (self._sublabs is None):
            self._subsets = self.__splitset__(self.data)
            self._sublabs = self.__splitset__(self.labels)

        if subset == 'train':
            self.subset = self._subsets[0]
            self.sublab = self._sublabs[0]
        elif subset == 'valid':
            self.subset = self._subsets[1]
            self.sublab = self._sublabs[1]
        elif subset == 'test':
            self.subset = self._subsets[2]
            self.sublab = self._sublabs[2]

        # Tensor, and type
        self.subset = self.__totensor__(self.subset)
        self.sublab = self.__totensor__(self.sublab)
        self.subset = self.subset.float()
        self.sublab = self.sublab.long()

        # Send to GPU
        if torch.cuda.is_available():
            self.subset = self.subset.cuda()
            self.sublab = self.sublab.cuda()

        return
    