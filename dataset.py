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

        # Data preprocessing stack
        self.preproc: list = None

        # Data subsets and labels
        self.subset: npt.ArrayLike = None
        self.sublab: npt.ArrayLike = None

        return

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple:
        return self.subset[idx], self.sublab[idx]
    
    def __totensor__(self, data: npt.ArrayLike) -> torch.TensorType:
        return torch.tensor(data)

    def __checkset__(self) -> bool:
        return (self.data.shape[0] == self.labels.shape[0])

    def choose_set(self) -> None:
        return
    
    def load_dataset(self, dataset: npt.ArrayLike) -> None:
        self.data = np.asarray(dataset, dtype=np.float32).copy()
        return

    def load_labels(self, labels: npt.ArrayLike) -> None:
        self.labels = np.asarray(labels, dtype=np.int32).copy()
        self.labels.reshape(-1, 1)
        return