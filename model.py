import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import multiprocessing as mp
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
        # stream: torch.cuda.Stream = None
    ) -> None:

        # Training progress output
        self.silent = silent

        # Dataset and id
        self.ds = ds
        self.id = mdl_id

        # Build DataLoader BEFORE any CUDA init (safe for num_workers>0)
        self.loader = torch.utils.data.DataLoader(
            dataset=self.ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        # self.loader = iter(self._loader)
        # self._num_batches = max(1, len(self._loader))

        # CUDA + model on its own stream
        torch.backends.cudnn.benchmark = True
        # self.stream = stream if stream is not None else torch.cuda.Stream()
        # with torch.cuda.stream(self.stream):
        self.model = torch.nn.Sequential(*layers).cuda()

        # Loss and optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Epoch and batch tracking
        self.num_epoch: int = num_epoch

        # Device-side loss accumulator (avoid per-step sync)
        self.running_loss = torch.zeros((), device="cuda", dtype=torch.float32)
        self.losses: npt.NDArray = np.zeros(shape=(self.num_epoch,))

        # Individual progress bar for this model
        self.pbar = None

    def __del__(self) -> None:
        if hasattr(self, 'pbar') and self.pbar is not None:
            self.pbar.close()
        # if self.stream is not None:
        #     self.stream.synchronize()
        return

    def train(self):

        if not self.silent:
            
            self.pbar = tqdm(
                total=self.num_epoch, 
                desc=f"Model {self.id}", 
                position=self.id + 1,
                leave=False,
                disable=self.silent
            )

        self.model.train()
        # with torch.cuda.stream(self.stream):
        for epoch in range(self.num_epoch):
            self.running_loss.zero_()
            for inputs, labels in self.loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.running_loss.add_(loss.detach())

            avg_loss = self.running_loss.item() / len(self.loader)
            self.losses[epoch] = avg_loss

            self.pbar.update(1)
            self.pbar.set_postfix({'loss': f"{avg_loss:.4f}"})

        return self.losses