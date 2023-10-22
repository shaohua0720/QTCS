import os
import sys
import time
import datetime
import argparse
import pickle
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.dataset_init_utils import get_dataset, get_dl_test_set
from utils.pytorchtools import EarlyStopping
from model.slvit_ae import *
from model.hybrid_vqvae import *

from utils.load_data_model import load_dataloader,load_model
from main_res_vae_quant_Train import output,nmse_eval
from utils.matrw import *

method = r'MGPU'

def main(config):
    print('Experimental Settings: \n')
    for arg in vars(config):
        print(arg, r':', getattr(config, arg))
    device = config.device
    foldername = config.foldername
    
    _, _, test_loader, dl_test_loader_1 = load_dataloader(batch_size=config.batch_size)
    model = load_model(config=config)

    if method == r'MGPU':
        snapshot = torch.load(config.snapshot_path)
        model.load_state_dict(snapshot["MODEL_STATE"])
    elif method == r'SGPU':
        snapshot_path = os.path.join(foldername,'checkpoint.pt')
        snapshot = torch.load(snapshot_path)
        model.load_state_dict(snapshot)
        
    loss_file = open(os.path.join(foldername, 'config.txt'), 'a+')

    # evaluate NMSE
    model.eval()
    # test loss
    mat_data = {}
    true_channels, pred_channels, Eout = output(model, test_loader)
    mat_data['Eout']=Eout.view(3000,-1).cpu().numpy()
    nmse_test, mse_test, nmse_ul = nmse_eval(true_channels, pred_channels)

    print('\nafter training:', file=loss_file)
    print('\ntest loss', file=loss_file)
    print('nmse = ' + str(nmse_test.item()), file=loss_file)
    print('mse = ' + str(mse_test.item()), file=loss_file)

    np.save(os.path.join(foldername, 'nmse_ul'), nmse_ul.cpu())

    model.eval()
    # evaluate nmse for DL test set
    true_channels, pred_channels, Eout = output(model, dl_test_loader_1)
    mat_data['Eout_dl']=Eout.view(3000,-1).cpu().numpy()
    scio.savemat(os.path.join(config.result_path,'mat_data.mat'),mat_data)
    
    nmse_test, mse_test, nmse_dl_1 = nmse_eval(true_channels, pred_channels)
    print('\n\n120 MHz frequency gap:', file=loss_file)
    print('\nafter training:', file=loss_file)
    print('\ntest loss', file=loss_file)
    print('nmse = ' + str(nmse_test.item()), file=loss_file)
    print('mse = ' + str(mse_test.item()), file=loss_file)

    np.save(os.path.join(foldername, 'nmse_dl_1'), nmse_dl_1.cpu())

    loss_file.close()

    # plot an example
    index = 10
    true_ex = true_channels[index, 0].detach().cpu().numpy()
    pred_ex = pred_channels[index, 0].detach().cpu().numpy()

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.imshow(true_ex)
    ax1.set_title('true channel')
    ax2.imshow(pred_ex)
    ax2.set_title('predicted channel')

    plt.tight_layout()
    plt.savefig(os.path.join(
        foldername, 'Real part channel before and after training.png'))

if __name__ == '__main__':
    """Run the script with the best configurations"""
    parser = argparse.ArgumentParser(description='MIMO CSI Compression')  
    parser.add_argument('--path', type = str, default = r'results/BGVAE/2023-10-20 09-20-09', metavar = 'F',
                        help = '')
    args = parser.parse_args()
    file = os.path.join(args.path,'config.pkl')
    with open(file,'rb') as f:
        config = pickle.load(f)
    config.foldername = args.path
    main(config=config)
