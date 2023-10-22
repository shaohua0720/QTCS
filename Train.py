import os
import sys
import time
import datetime
import argparse
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.pytorchtools import EarlyStopping
from utils.config import Config
from utils.loader import *
from models import HybridNet,QCSLoss
from utils.pytorchtools import set_seed


# device = torch.device('cuda:0')
torch.autograd.set_detect_anomaly(True)
def train_epoch(model, optimizer, train_loader, config):
    device = config.device
    total_samples = len(train_loader.dataset)
    model.train()

    # i = 0
    loss = []
    for i, data in enumerate(train_loader):
        i = i +1
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data)
        loss = QCSLoss(output,data)
            
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            print(
                '[' + '{:5}'.format(i * len(data)) + '/' +
                '{:5}'.format(total_samples)
                + ' (' + '{:3.0f}'.format(100 * i /
                                          len(train_loader)) + '%)]  train loss: '
                + format(
                    loss.item(), '.3f'
                )
            )

def output(model, data_loader,config):
    device = config.device
    model.eval()
    with torch.no_grad():
        i = 0
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            if i == 0:
                pre = model(data)
                true = target
            else:
                pre = torch.cat((pre, model(data)), dim=0)
                true = torch.cat((true, target), dim=0)
            i += 1
        return true, pre

def evaluate(model, data_loader):
    true, pre,_ = output(model, data_loader)
    nmse, mse, nmse_all = nmse_eval(true, pre)
    
    print('\nvalidation loss (mse): ' + format(mse.item(), '.3f'))
    print('validation loss (nmse): ' + format(nmse.item(), '.3f'))
    print('----------------------------------------------------\n')
    return nmse, mse

def nmse_eval(y_true, y_pred):
    y_true = y_true.reshape(len(y_true), -1)
    y_pred = y_pred.reshape(len(y_pred), -1)

    mse = torch.sum(abs(y_pred - y_true) ** 2, dim=1)
    power = torch.sum(abs(y_true) ** 2, dim=1)

    nmse = mse / power
    return torch.mean(nmse), torch.mean(mse), nmse

def main(config):
    config.check()
    train_loader, val_loader, test_loader = get_loader(config)
    model = HybridNet(config).to(config.device)        
    logs = SummaryWriter(config.folder)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=5, verbose=True)

    # train the model
    reco_loss_history = []
    commit_loss_history = []

    es = EarlyStopping(patience=10, verbose=False, delta=0.001, path=config.model)

    n_epochs = config.epochs

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        print('Epoch:', epoch)

        train_epoch(model, optimizer, train_loader, config)

        val_loss_nmse, val_loss_mse = evaluate(model, config)
        logs.add_scalar('validate MSE(dB)',10*np.log10(val_loss_mse.cpu()),epoch)
        scheduler.step(val_loss_mse)

        # early stopping
        es(val_loss_mse, model)
        if es.early_stop:
            print('early stopping')
            break

    print('Execution time:', '{:5.2f}'.format(
        time.time() - start_time), 'seconds')
    print('--'*20)

    model.load_state_dict(torch.load(config.model))
    loss_file = open(os.path.join(config.folder, 'config.txt'), 'a+')

    # evaluate NMSE
    model.eval()
    true_channels, pred_channels,_ = output(model, test_loader)
    nmse_test, mse_test, nmse_ul = nmse_eval(true_channels, pred_channels)
    print('\nafter training:', file=loss_file)
    print('\ntest loss', file=loss_file)
    print('nmse = ' + str(nmse_test.item()), file=loss_file)
    print('mse = ' + str(mse_test.item()), file=loss_file)
    np.save(os.path.join(config.folder, 'nmse_ul'), nmse_ul.cpu())
    loss_file.close()

if __name__ == '__main__':
    """Run the script with the best configurations"""
    parser = argparse.ArgumentParser(description='MIMO CSI Compression')  
    # parser.add_argument('--network', type = str, default = 'BGVAE', metavar = 'N',
    #                     help = 'Deep learning model [NDVitResVAE, NDVitAE, SLViTAEQuant, SLViTAE, BGVAE]')
    # parser.add_argument('--patch_sizes_idx', type = int, default = 7, metavar = 'PS',
    #                     help = 'patch size for ViT')
    # parser.add_argument('--dims_enc', type=int, nargs='+', default = [128, 64, 32], metavar = 'EM',
    #                     help = 'Encoder dimensions') 
    # parser.add_argument('--h1h2h3', type=int, nargs='+', default = [16, 16, 16], metavar = 'H',
    #                     help = 'Downsampling')   
    # parser.add_argument('--hidden_dim', type = int, default = 64, metavar = 'MH',
    #                     help = 'Hiddent dimension of the MLP in ViT')
    parser.add_argument('--batch_size', type = int, default = 25, metavar = 'B',
                        help = 'Batch size of the training')
    # parser.add_argument('--mode', type = str, default = 'linear', metavar = 'M',
    #                     help = 'Upsampling mode')
    parser.add_argument('--lr', type = float, default = 0.006015353710001343, metavar = 'L',
                        help = 'Learning rate')
    # parser.add_argument('--beta', type = float, default = 0.2780677050121794, metavar = 'B',
    #                     help = 'Coefficients of VAVQE')
    parser.add_argument('--epochs', type = int, default = 100, metavar = 'E',
                        help = 'numer of Epochs')
    # parser.add_argument('--num_embeddings', type = int, default = 64, metavar = 'NE',
    #                     help = 'Codebook size of VQVAE')
    # parser.add_argument('--embeddings_dim', type = int, default = 1, metavar = 'ED',
    #                     help = 'Codebook dimension of VQVAE')
    parser.add_argument('--device', type = str, default = 'cuda:0', metavar = 'DV',
                        help = 'GPU')
    # parser.add_argument('--succ_prob', type = float, default = 1/512, metavar = 'P',
    #                     help = 'Probability of Geometric distribution')
    # parser.add_argument('--nd_enable', type = int, default = 1, metavar = 'P',
    #                     help = 'Nested drop is enabled')
    args = parser.parse_args()
    
    config = Config()
    config.device= args.device
    config.epochs=args.epochs
    config.batch_size=args.batch_size
    config.lr = args.lr

    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)
    set_seed(123456)
    main(config)
