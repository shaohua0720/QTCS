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
    train_dataloader, _, _, _ =  g(args.batch_size)
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
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', type=int, default= 51, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, default=10, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=25, type=int, help='Input batch size on each device (default: 23)')
    parser.add_argument('--network', type = str, default = 'BGVAE', metavar = 'N',
                        help = 'Deep learning model [NDVitResVAE, NDVitAE, SLViTAEQuant, SLViTAE,BGVAE]')
    parser.add_argument('--patch_sizes_idx', type = int, default = 6, metavar = 'PS',
                        help = 'patch size for ViT')
    parser.add_argument('--dims_enc', type=int, nargs='+', default = [128, 64, 32], metavar = 'EM',
                        help = 'Encoder dimensions') 
    parser.add_argument('--h1h2h3', type=int, nargs='+', default = [16, 16, 16], metavar = 'H',
                        help = 'Downsampling')   
    parser.add_argument('--hidden_dim', type = int, default = 64, metavar = 'MH',
                        help = 'Hiddent dimension of the MLP in ViT')
    parser.add_argument('--mode', type = str, default = 'linear', metavar = 'M',
                        help = 'Upsampling mode')
    parser.add_argument('--lr', type = float, default = 0.006, metavar = 'L',
                        help = 'Learning rate')
    parser.add_argument('--beta', type = float, default = 0.28, metavar = 'B',
                        help = 'Coefficients of VAVQE')
    parser.add_argument('--decay', type = float, default = 0, metavar = 'D',
                        help = 'EMA mode')
    parser.add_argument('--num_embeddings', type = int, default = 64, metavar = 'NE',
                        help = 'Codebook size of VQVAE')
    parser.add_argument('--embeddings_dim', type = int, default = 1, metavar = 'ED',
                        help = 'Codebook dimension of VQVAE')
    parser.add_argument('--device', type = str, default = 'cuda:0', metavar = 'DV',
                        help = 'GPU')
    parser.add_argument('--succ_prob', type = float, default = 1/512, metavar = 'P',
                        help = 'Probability of Geometric distribution')
    parser.add_argument('--nd_enable', type = int, default = 1, metavar = 'P',
                        help = 'Nested drop is enabled')
    args = parser.parse_args()
    
    main(args)
