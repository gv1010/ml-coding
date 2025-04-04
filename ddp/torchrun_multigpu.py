
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup():
    """
    rank: unique id of eeach process
    world_size: no of gpus or devices
    """
    
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(10), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]

class Trainer:
    def __init__(self, model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path) -> None:

        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.optimizer = optimizer
        self.train_data = train_data
        self.save_every = save_every
        
        self.snapshot_path = snapshot_path

        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(model, device_ids = [self.gpu_id])

    
    def _load_snapshot(self, snapshot_path):
        device_loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=device_loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

        
    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()


    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_data))[0])
        print(f"[GPU {self.gpu_id}] Epoch {epoch} | \
                Batchsize: {batch_size} | Steps: {len(self.train_data)}")

        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        check_point = self.model.state_dict()
        path = "checkpoint.pt"
        torch.save(check_point, path)
        print(f"Epoch {epoch} | Training checkpoint saved at {path}")

    def train(self, max_epochs):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        

def load_train_objs():
    train_set = MyTrainDataset(20248)
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)
    return train_set, model, optimizer

def prepare_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, sampler = DistributedSampler(dataset))

def main(total_epochs, save_every, batch_size, snapshot_path = "snapshot.pt"):

    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)

if __name__ == "__main__":
    main(20, 5, 16)
