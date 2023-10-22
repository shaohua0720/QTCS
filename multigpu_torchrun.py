# Original Code template from Pytorch tutorial
# https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py
import torch
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import datetime
import platform

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from models import HybridNet,QCSLoss
from utils.config import *
from utils.loader import *
from utils.args import Option

sys=''
torch.set_default_dtype(torch.float32)
if platform.system().lower() == 'windows':
    sys=r'win'
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # os.environ["OMP_NUM_THREADS"]=1

def ddp_setup():
    if sys==r'win':
        init_process_group(backend="gloo")
    else:
        init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id],find_unused_parameters=True)

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        # output = self.model(source)
        # data_reco, _, loss_t = self.model(source)
        # reco_loss = F.mse_loss(data_reco, source)
        # loss = reco_loss + loss_t
        data_reco, _  = self.model(source)
        loss = QCSLoss(data_reco,targets)
                        
        # loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs(args):
    model = HybridNet(args)
    train_dataloader, _, _ =  get_loader(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=5, verbose=True)
    return train_dataloader, model, optimizer, scheduler


def gen_path(args):
    now = datetime.datetime.now()
    rs_path = os.path.join(r'results',args.network,now.strftime("%Y-%m-%d %H-%M-%S"))
    snapshot_path = os.path.join(rs_path,'snapshot.pt')
    setattr(args,'result_path',rs_path)
    setattr(args,'snapshot_path',snapshot_path)
    
def save_config(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    print('Experimental Settings \n')
    with open(os.path.join(args.result_path, 'config.txt'), 'w') as f:
        for arg in vars(args):
            print(arg, r':', getattr(args, arg))
            print(arg, getattr(args, arg),file=f)
    with open(os.path.join(args.result_path, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)
        
def main(args):
    ddp_setup()
    gpu_id = int(os.environ["LOCAL_RANK"])
    gen_path(args)
    if gpu_id==0:
        save_config(args)
    train_data, model, optimizer, scheduler = load_train_objs(args)
    trainer = Trainer(model, train_data, optimizer, args.save_every, args.snapshot_path)
    trainer.train(args.total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    config = Option().parse()
    main(config)
