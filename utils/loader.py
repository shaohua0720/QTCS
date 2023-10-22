import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import h5py
from config import Config

class ChannelData(Dataset):
    def __init__(self, filepath,key_name) -> None:
        CHL = h5py.File(filepath)
        CHL_R = np.real(CHL[key_name])
        CHL_I = np.imag(CHL[key_name])
        self.data = np.concatenate([CHL_R,CHL_I],axis=1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index,:,:]

def get_loader(config):
    train_set = ChannelData(config.train_data,'train')
    val_set = ChannelData(config.val_data,'val')
    test_set = ChannelData(config.test_data,'test')
    if config.DDP:
        train_loader = DataLoader(train_set,batch_size=config.batch_size, \
                                  shuffle=False,pin_memory=True,sampler=DistributedSampler(train_set))
        val_loader   = DataLoader(val_set,batch_size=config.batch_size,\
                                shuffle=False,pin_memory=True,sampler=DistributedSampler(train_set))
        test_loader  = DataLoader(test_set,batch_size=config.batch_size,\
                                shuffle=False,pin_memory=True,sampler=DistributedSampler(train_set))
    else:
        train_loader = DataLoader(train_set,batch_size=config.batch_size,shuffle=True,pin_memory=True)
        val_loader   = DataLoader(val_set,batch_size=config.batch_size,shuffle=False)
        test_loader  = DataLoader(test_set,batch_size=config.batch_size,shuffle=False)
    return (train_loader,val_loader,test_loader)

if __name__ == '__main__':
    my_config = Config()
    train,val,test = get_loader(my_config)
    print(train.shape)

    
