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
from utils.args import Option

# device = torch.device('cuda:0')
torch.autograd.set_detect_anomaly(True)
def train_epoch(model, optimizer, train_loader, config):
    device = config.device
    total_samples = len(train_loader.dataset)
    model.train()

    i = 0
    loss = []
    for data, target in train_loader:
        i = i +1
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output,quant_loss,commit_loss = model(data)
        loss = QCSLoss(target,output,quant_loss,commit_loss)
            
        loss.backward()
        optimizer.step()

        # if i % 10 == 0:
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
                pre = model(data)[0]
                true = target
            else:
                pre = torch.cat((pre, model(data)[0]), dim=0)
                true = torch.cat((true, target), dim=0)
            i += 1
        return true, pre

def evaluate(model, data_loader, config):
    true, pre = output(model, data_loader,config)
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

def _load_snapshot(model, snapshot_path):
        snapshot = torch.load(snapshot_path)
        model.load_state_dict(snapshot["MODEL_STATE"])
        epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {epochs_run}")
        return epochs_run

def _save_snapshot(model, config, epoch):
        snapshot = {
            "MODEL_STATE": model.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, config.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {config.snapshot_path}")

def main(config):
    train_loader, val_loader, test_loader = get_loader(config)
    model = HybridNet(config).to(config.device)        
    logs = SummaryWriter(config.folder)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=5, verbose=True)

    es = EarlyStopping(patience=10, verbose=False, delta=0.001, path=config.model)

    n_epochs = config.epochs
    start_time = time.time()

    start_epoch = 1
    if config.resume and os.path.exists(config.snapshot_path):
        start_epoch=_load_snapshot(model,config.snapshot_path)

    for epoch in range(start_epoch, n_epochs + 1):
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_loader, config)
        # update scheduler
        val_loss_nmse, val_loss_mse = evaluate(model, val_loader, config)
        logs.add_scalar('validate MSE(dB)',10*np.log10(val_loss_mse.cpu()),epoch)
        scheduler.step(val_loss_mse)

        if epoch % config.save_every == 0:
            _save_snapshot(model,config,epoch)

        # early stopping
        es(val_loss_mse, model)
        if es.early_stop:
            print('early stopping')
            break

    print('Execution time:', '{:5.2f}'.format(
        time.time() - start_time), 'seconds')
    print('--'*20)

    # evaluate NMSE
    model.load_state_dict(torch.load(config.model))
    rs_file = open(os.path.join(config.folder, 'config.txt'), 'a+')
    model.eval()
    true_channels, pred_channels = output(model, test_loader,config)
    nmse_test, mse_test, nmse_ul = nmse_eval(true_channels, pred_channels)
    print('\nafter training:', file=rs_file)
    print('\ntest loss', file=rs_file)
    print('nmse = ' + str(nmse_test.item()), file=rs_file)
    print('mse = ' + str(mse_test.item()), file=rs_file)
    rs_file.close()
    np.save(os.path.join(config.folder, 'nmse_ul'), nmse_ul.cpu())
    

if __name__ == '__main__':
    
    config = Option().parse()
    config.check()
    config.show()
    torch.cuda.empty_cache()
    # torch.autograd.set_detect_anomaly(True)

    set_seed(123456)
    main(config)
