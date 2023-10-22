import numpy as np
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import h5py
import torch
from .config import Config
import os
os.environ['PYDEVD_WARN_EVALUATION_TIMEOUT']='120'

def channelData(filepath):
    with h5py.File(filepath,'r') as CHL:
        CHL_np =np.real(CHL['/data'])
        return torch.from_numpy(CHL_np.astype(np.float32))

def get_loader(config):
    train_set = channelData(config.train_data)
    val_set = channelData(config.val_data)
    test_set = channelData(config.test_data)
    trainset = TensorDataset(train_set,train_set)
    valset = TensorDataset(val_set,val_set)
    testset = TensorDataset(test_set,test_set)

    if config.DDP:
        train_loader = DataLoader(trainset,batch_size=config.batch_size, \
                                  shuffle=False,pin_memory=True,sampler=DistributedSampler(train_set))
        val_loader   = DataLoader(valset,batch_size=config.batch_size,\
                                shuffle=False,pin_memory=True,sampler=DistributedSampler(train_set))
        test_loader  = DataLoader(testset,batch_size=config.batch_size,\
                                shuffle=False,pin_memory=True,sampler=DistributedSampler(train_set))
    else:
        train_loader = DataLoader(trainset,batch_size=config.batch_size,shuffle=True,pin_memory=True)
        val_loader   = DataLoader(valset,batch_size=config.batch_size,shuffle=False)
        test_loader  = DataLoader(testset,batch_size=config.batch_size,shuffle=False)
    return (train_loader,val_loader,test_loader)

if __name__ == '__main__':
    my_config = Config()
    train,val,test = get_loader(my_config)
    print(train.shape)

    
