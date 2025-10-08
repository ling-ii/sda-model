import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import threading
import torch

class Model:
    def __init__(
        self,
        layers,
        ds,
        learning_rate:float=0.001,
        batch_size:int=1000,
        num_epoch:int=500,
        mdl_id: int=0,
        silent: bool=False,
        stream: torch.cuda.Stream=None
    ) -> None:
        
        # Training progress output
        self.silent = silent

        # Ensure model on new stream
        self.stream = stream if stream is not None else torch.cuda.Stream()
        with torch.cuda.stream(self.stream):
            self.model = torch.nn.Sequential(*layers).cuda()
        self.stream.synchronize()

        # Model dataset and id
        self.ds = ds
        self.id = mdl_id

        # Loss criterion and gradient optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate
        )

        # Dataloader
        self.loader = torch.utils.data.DataLoader(
            dataset=self.ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Training data
        self.num_epoch: int = num_epoch
        self.losses: npt.NDArray = np.zeros(shape=(self.num_epoch,))

    def __del__(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()
        return

    def __train__(self) -> npt.NDArray:

        with tqdm(
            total=self.num_epoch,
            desc=f"Training model {self.id}",
            leave=False,
            unit="epoch",
            position=self.id+1 # Hacky way of maintaining multiple pbar
        ) as pbar:
            
            for epoch in range(self.num_epoch):
                self.model.train()
                running_loss: float = 0

                # Training loop
                for inputs, labels in self.loader:
                    # Set cuda stream
                    with torch.cuda.stream(self.stream):
                        # Ensure data is on correct stream
                        inputs = inputs.cuda(non_blocking=True)
                        labels = labels.cuda(non_blocking=True)

                        self.optimizer.zero_grad()
                        outputs = self.model(inputs)
                        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2).float()
                        loss = self.criterion(outputs, one_hot_labels)
                        loss.backward()
                        self.optimizer.step()
                        running_loss += loss.item()

                # Update losses
                avg_loss = running_loss / len(self.loader)
                self.losses[epoch] = avg_loss

                # Update progress
                pbar.set_postfix(loss=f"{avg_loss:.4f}")
                pbar.update(1)

        # Release GPU resources
        self.model.cpu()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return self.losses
    
    def __silenttrain__(self) -> npt.NDArray:
        for epoch in range(self.num_epoch):
            self.model.train()
            running_loss: float = 0

            # Training loop
            for inputs, labels in self.loader:
                # Set cuda stream
                with torch.cuda.stream(self.stream):
                    # Ensure data is on correct stream
                    inputs = inputs.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2).float()
                    loss = self.criterion(outputs, one_hot_labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()

            # Update losses
            avg_loss = running_loss / len(self.loader)
            self.losses[epoch] = avg_loss
        
        # Release GPU resources
        self.model.cpu()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return self.losses
    
    def train(self) -> npt.NDArray:
        return (self.__silenttrain__() if self.silent else self.__train__())
    

class MultiTrainer:
    def __init__(self, mdls: list[Model]) -> None:
        self.mdls = mdls
        self.results = {}
        self.threads = []
        self.streams = [mdl.stream for mdl in mdls]
        return

    def __trainmodel__(self, mdl: Model) -> None:
        """Wrapper for Model.train()"""
        self.results[mdl.id] = mdl.train()
        return
    
    def train_models(self) -> dict:
        # Start new thread for each model
        for mdl in self.mdls:
            thread = threading.Thread(
                target=self.__trainmodel__,
                args=(mdl,)
            )
            self.threads.append(thread)
            thread.start()

        # Await threads
        for thread in self.threads:
            thread.join()

        # Synchronize streams
        for stream in self.streams:
            stream.synchronize()

        return self.results
    