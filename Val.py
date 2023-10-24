import os
import sys
import time
import datetime
import argparse
import pickle
import torch
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.loader import get_loader
from utils.pytorchtools import EarlyStopping
from models.demo import HybridNet,QCSLoss

# from utils.load_data_model import load_dataloader,load_model
from Train import output,nmse_eval
# from utils.matrw import *

method = r'SGPU'

def main(config):
    print('Experimental Settings: \n')
    config.show()
    
    device = config.device
    foldername = config.folder
    
    _, _, test_loader = get_loader(config)
    model = HybridNet(config).to(device)

    if method == r'MGPU':
        snapshot = torch.load(config.model)
        model.load_state_dict(snapshot["MODEL_STATE"])
    elif method == r'SGPU':
        snapshot = torch.load(config.model)
        model.load_state_dict(snapshot)
        
    # evaluate NMSE
    rs_file = open(os.path.join(foldername, 'config.txt'), 'a+')
    model.eval()
    true_channels, pred_channels = output(model, test_loader, config)
    nmse_test, mse_test, nmse_ul = nmse_eval(true_channels, pred_channels)

    print('\nafter training:', file=rs_file)
    print('\ntest loss', file=rs_file)
    print('nmse = ' + str(nmse_test.item()), file=rs_file)
    print('mse = ' + str(mse_test.item()), file=rs_file)
    rs_file.close()

    np.save(os.path.join(foldername, 'nmse_ul'), nmse_ul.cpu())


if __name__ == '__main__':
    """Run the script with the best configurations"""
    parser = argparse.ArgumentParser(description='MIMO CSI Compression')  
    parser.add_argument('--path', type = str, default = r'results/QCSMIMO/10/models', metavar = 'F',
                        help = '')
    args = parser.parse_args()
    file = os.path.join(args.path,'config.pkl')
    with open(file,'rb') as f:
        config = pickle.load(f)
    config.foldername = args.path
    main(config=config)
