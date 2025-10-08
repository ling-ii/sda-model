import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import multiprocessing as mp
import threading
import torch

class Model:
    def __init__(
        self,
        layers,
        ds,
        learning_rate: float = 0.001,
        batch_size: int = 1000,
        num_epoch: int = 500,
        mdl_id: int = 0,
        silent: bool = False,
        stream: torch.cuda.Stream = None
    ) -> None:

        # Training progress output
        self.silent = silent

        # Dataset and id
        self.ds = ds
        self.id = mdl_id

        # Build DataLoader BEFORE any CUDA init (safe for num_workers>0)
        self._loader = torch.utils.data.DataLoader(
            dataset=self.ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,           # avoid zero-batch epochs
            # pin_memory=True,           # enables async H2D
            num_workers=0,             # overlap CPU loading with GPU work
            # persistent_workers=True,
            # prefetch_factor=2,
        )
        self.loader = iter(self._loader)
        self._num_batches = max(1, len(self._loader))

        # CUDA + model on its own stream
        torch.backends.cudnn.benchmark = True
        self.stream = stream if stream is not None else torch.cuda.Stream()
        with torch.cuda.stream(self.stream):
            self.model = torch.nn.Sequential(*layers).cuda()

        # Loss and optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Epoch and batch tracking
        self.num_epoch: int = num_epoch
        self.epoch_counter: int = 0
        self.batch_counter: int = 0
        self.total_batches_processed: int = 0

        # Device-side loss accumulator (avoid per-step sync)
        self.running_loss = torch.zeros((), device="cuda", dtype=torch.float32)
        self.losses: npt.NDArray = np.zeros(shape=(self.num_epoch,))

        # Individual progress bar for this model
        self.pbar = tqdm(
            total=self.num_epoch, 
            desc=f"Model {self.id}", 
            position=self.id + 1,
            leave=True,
            disable=self.silent
        )

    def __del__(self) -> None:
        if hasattr(self, 'pbar') and self.pbar is not None:
            self.pbar.close()
        if self.stream is not None:
            self.stream.synchronize()
        return
