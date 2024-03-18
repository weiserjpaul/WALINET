import numpy as np
import h5py
import time
import os
import sys

import torch
import torch.nn as nn


from src.model import yModel



def prediction(params, model):
    
    device = torch.device(params["gpu"] if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    ### Load Data ###
    path = params["path_to_data"] + params["val_subjects"][0] + '/CNNLipTrainingData_' + params["val_subjects"][0] + '.h5'
    fh = h5py.File(path, 'r')

    lip = torch.tensor(np.array(fh['test']['spectra'][:]))
    lipProj = torch.tensor(np.array(fh['test']['Lipid_BL_Wat_spectra'][:]))

    spectra_energy = torch.sqrt(torch.sum(torch.abs(lip-lipProj)**2, dim=1))[:,None]
    lip /= spectra_energy
    lip = torch.stack((torch.real(lip), torch.imag(lip)), axis=1)
    lipProj /= spectra_energy
    lipProj = torch.stack((torch.real(lipProj), torch.imag(lipProj)), axis=1)

    prediction = lipidRemoval(lip, lipProj, model, device)
    prediction = prediction*spectra_energy

    hf = h5py.File(params["path_to_model"] + 'predictions/' + params["val_subjects"][0] + '.h5', 'w')
    hf.create_dataset('pred', data=prediction)
    hf.close()

def lipidRemoval(lip, lipProj, model, device):

    ### Lipid Removal ###
    datasz = lipProj.shape[0]
    batchsz = 200
    sta_epoch = time.time()
    pred=None
    prediction = torch.zeros((lipProj.shape[0],400), dtype=torch.cfloat)
    model.eval()
    with torch.no_grad():
        for i in range(int(datasz/batchsz)):
            log = 'Percent: {:.2f}%'
            percent = (i+1)/int(datasz/batchsz)*100
            print(log.format(percent), end='\r')
            lip_batch = lip[i*batchsz:(i+1)*batchsz,:,:]
            lipProj_batch = lipProj[i*batchsz:(i+1)*batchsz,:,:]
            
            lip_batch, lipProj_batch = lip_batch.to(device), lipProj_batch.to(device)
            pred = model(lip_batch, lipProj_batch).cpu()
            prediction[i*batchsz:(i+1)*batchsz,:] = pred[:,0] + 1j*pred[:,1]

    sto_epoch = time.time() - sta_epoch
    log_epoch = 'Lipid Removal: Time: {:.4f}'
    print(log_epoch.format(sto_epoch))

    return prediction


