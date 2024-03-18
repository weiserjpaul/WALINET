import numpy as np
import h5py
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



def training(model, params, dataloader, device, epoch):
    model.train()
    epoch_loss = []
    epoch_lossconv = []
    sta_epoch = time.time()
    if params["n_batches"] == -1 or params["n_batches"] > len(dataloader):
        n_batches_tmp = len(dataloader)
    else:
        n_batches_tmp = params["n_batches"]
    
    for i,batch in enumerate(dataloader):
        if i >= n_batches_tmp:
            break
        sta_batch = time.time()
        
        spectra_All, spectra_IDLip, spectra_Lip = batch
        spectra_All, spectra_IDLip, spectra_Lip = spectra_All.to(device), spectra_IDLip.to(device), spectra_Lip.to(device)
        
        pred = model(spectra_All, spectra_IDLip)

        loss = params["loss_func"](pred, spectra_Lip)
        loss.backward()
        params["optimizer"].step()
        params["optimizer"].zero_grad()
        
        epoch_loss.append(loss.cpu().item())
        sto_batch = time.time() - sta_batch
        if params["verbose"]:
            log_batch = ' ~ Epoch: {:03d}, Batch: ({:03d}/{:03d}) Loss: {:.8f}, GPU-Time: {:.4f}'
            print(log_batch.format(epoch+1, i+1, n_batches_tmp, loss.item(), sto_batch))
        
    epoch_loss = np.mean(np.array(epoch_loss))
    sto_epoch = time.time() - sta_epoch
    log_epoch = 'Epoch: {:03d}, Loss: {:.8f}, Time: {:.4f}'
    print(log_epoch.format(epoch+1, epoch_loss , sto_epoch))
    return model, epoch_loss
    
def validation(model, params, dataloader, device, epoch):
    model.eval()
    epoch_loss = []
    sta_epoch = time.time()
    if params["n_val_batches"] == -1 or params["n_val_batches"] > len(dataloader):
        n_batches_tmp = len(dataloader)
    else:
        n_batches_tmp = params["n_val_batches"]
    
    for i,batch in enumerate(dataloader):
        if i >= n_batches_tmp:
            break
        sta_batch = time.time()
        
        spectra_All, spectra_IDLip, spectra_Lip = batch
        spectra_All, spectra_IDLip, spectra_Lip = spectra_All.to(device), spectra_IDLip.to(device), spectra_Lip.to(device)
        
        pred = model(spectra_All, spectra_IDLip)

        loss = params["loss_func"](pred, spectra_Lip)
        
        epoch_loss.append(loss.cpu().item())
        sto_batch = time.time() - sta_batch
        if params["verbose"]:
            log_batch = ' ~ ValEpoch: {:03d}, Batch: ({:03d}/{:03d}) Loss: {:.8f}, GPU-Time: {:.4f}'
            print(log_batch.format(epoch+1, i+1, n_batches_tmp, loss.item(), sto_batch))
        
    epoch_loss = np.mean(np.array(epoch_loss))
    sto_epoch = time.time() - sta_epoch
    log_epoch = 'ValEpoch: {:03d}, Loss: {:.8f}, Time: {:.4f}'
    print(log_epoch.format(epoch+1, epoch_loss , sto_epoch))
    return epoch_loss