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
from utils.load_data_model import load_dataloader,load_model
from model.bg_vae import bgvae_loss
from utils.train_utils import *

device = torch.device('cuda:0')
torch.autograd.set_detect_anomaly(True)
def train_epoch(model, optimizer, data_loader, reco_loss_history, commit_loss_history, beta=0.25, ema=False):
    total_samples = len(data_loader.dataset)
    model.train()

    running_reco_loss = 0.0
    running_commit_loss = 0.0
    i = 0
    loss = []
    for data, target in data_loader:
        i = i +1
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        if model.name =='NDVitResVAE':
            data_reco, _, loss_t = model(data)
            reco_loss = F.mse_loss(data_reco, data)
            loss = reco_loss + loss_t
        elif model.name == 'NDVitAE':
            data_reco, _= model(data)
            loss = F.mse_loss(data_reco,data)
        elif model.name == 'SLViTAEQuant':
            if ema is not True:
                data_reco, quant_loss, commit_loss, perplexity = model(data)
            else:
                data_reco, commit_loss, perplexity = model(data)
                quant_loss = 0.0
            reco_loss = F.mse_loss(data_reco, data)
            loss = reco_loss + quant_loss + beta * commit_loss
        elif model.name == 'SLViTAE':
            output,_ = model(data)
            loss = F.mse_loss(output, target, reduction="sum")
            loss = loss / data.shape[0]
        elif model.name == 'BGVAE':
            output, _,  = model(data)
            loss = bgvae_loss(output,data)
        else:
            raise NotImplementedError
            
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            print(
                '[' + '{:5}'.format(i * len(data)) + '/' +
                '{:5}'.format(total_samples)
                + ' (' + '{:3.0f}'.format(100 * i /
                                          len(data_loader)) + '%)]  train loss: '
                + format(
                    loss.item(), '.3f'
                )
            )

    reco_loss_history.append(running_reco_loss / len(data_loader))
    commit_loss_history.append(running_commit_loss / len(data_loader))

def output(model, data_loader):
    model.eval()
    with torch.no_grad():
        i = 0
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            if i == 0:
                pre = model(data)[0]
                true = target
                Eout = model(data)[1]
            else:
                pre = torch.cat((pre, model(data)[0]), dim=0)
                true = torch.cat((true, target), dim=0)
                Eout = torch.cat((Eout,model(data)[1]),dim=0)
            i += 1
        return true, pre, Eout

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
    device = config.device
    train_loader, validation_loader, test_loader, dl_test_loader_1 = load_dataloader(batch_size=config.batch_size)
    model = load_model(config=config)

    # Create result sub-folder
    now = datetime.datetime.now()
    foldername = os.path.join(r'results', config.network, now.strftime("%Y-%m-%d %H-%M-%S"))
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    print('Experimental Settings \n')
    with open(os.path.join(foldername, 'config.txt'), 'w') as f:
        for arg in vars(config):
            print(arg, r':', getattr(config, arg))
            print(arg, getattr(config, arg),file=f)
    with open(os.path.join(foldername, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)
        
    logs = SummaryWriter(foldername) # monitoring
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=5, verbose=True)

    # train the model
    reco_loss_history = []
    commit_loss_history = []

    es = EarlyStopping(patience=10, verbose=False, delta=0.001,
                       path=os.path.join(foldername, 'checkpoint.pt'))

    n_epochs = 1000
    if config.decay > 0.0:
        ema = True
    else:
        ema = False

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        print('Epoch:', epoch)

        train_epoch(
            model, optimizer, train_loader, reco_loss_history, commit_loss_history, config.beta,
            ema
        )

        val_loss_nmse, val_loss_mse = evaluate(model, validation_loader)
        logs.add_scalar('validate MSE(dB)',10*np.log10(val_loss_mse.cpu()),epoch)
        scheduler.step(val_loss_mse)

        # early stopping
        es(val_loss_mse, model)
        if es.early_stop:
            print('early stopping')
            break

    print('Execution time:', '{:5.2f}'.format(
        time.time() - start_time), 'seconds')
    print('----------------------------------------------------')

    model.load_state_dict(torch.load(
        os.path.join(foldername, 'checkpoint.pt')))

    loss_file = open(os.path.join(foldername, 'config.txt'), 'a+')

    # evaluate NMSE
    model.eval()
    # test loss
    true_channels, pred_channels,_ = output(model, test_loader)
    nmse_test, mse_test, nmse_ul = nmse_eval(true_channels, pred_channels)

    print('\nafter training:', file=loss_file)
    print('\ntest loss', file=loss_file)
    print('nmse = ' + str(nmse_test.item()), file=loss_file)
    print('mse = ' + str(mse_test.item()), file=loss_file)

    np.save(os.path.join(foldername, 'nmse_ul'), nmse_ul.cpu())

    model.eval()
    # evaluate nmse for DL test set
    true_channels, pred_channels,_ = output(model, dl_test_loader_1)
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

    # learning curves
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(reco_loss_history, label='reco loss')
    ax1.plot(commit_loss_history, label='commit loss')

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    title = 'model loss'
    ax1.set_title(title)
    ax1.legend(loc='best')

    plt.savefig(os.path.join(foldername, 'learning_curves.png'))

if __name__ == '__main__':
    """Run the script with the best configurations"""
    parser = argparse.ArgumentParser(description='MIMO CSI Compression')  
    parser.add_argument('--network', type = str, default = 'BGVAE', metavar = 'N',
                        help = 'Deep learning model [NDVitResVAE, NDVitAE, SLViTAEQuant, SLViTAE, BGVAE]')
    parser.add_argument('--patch_sizes_idx', type = int, default = 7, metavar = 'PS',
                        help = 'patch size for ViT')
    parser.add_argument('--dims_enc', type=int, nargs='+', default = [128, 64, 32], metavar = 'EM',
                        help = 'Encoder dimensions') 
    parser.add_argument('--h1h2h3', type=int, nargs='+', default = [16, 16, 16], metavar = 'H',
                        help = 'Downsampling')   
    parser.add_argument('--hidden_dim', type = int, default = 64, metavar = 'MH',
                        help = 'Hiddent dimension of the MLP in ViT')
    parser.add_argument('--batch_size', type = int, default = 25, metavar = 'B',
                        help = 'Batch size of the training')
    parser.add_argument('--mode', type = str, default = 'linear', metavar = 'M',
                        help = 'Upsampling mode')
    parser.add_argument('--lr', type = float, default = 0.006015353710001343, metavar = 'L',
                        help = 'Learning rate')
    parser.add_argument('--beta', type = float, default = 0.2780677050121794, metavar = 'B',
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
    
    set_seed(123456)
    main(args)
