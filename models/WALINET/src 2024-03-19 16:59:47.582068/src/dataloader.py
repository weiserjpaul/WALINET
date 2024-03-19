import numpy as np
import h5py
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SpectrumDatasetLoad(Dataset):
    def __init__(self, params, files):
        self.params = params
        self.files = files
        self.all_acc = ["2"]
        self.tag='train'

        print("Loading Data:")
        for i, f in enumerate(self.files):
            acc=self.all_acc[0]
            print("++ " + f)
            path = self.params["path_to_data"] + f + '/CNNLipTrainingData_' + f + '.h5'
            fh = h5py.File(path, 'r')
            
            spectra_Lip = torch.tensor(np.array(fh[self.tag]['Lipid_BL_Wat_spectra'][:]))
            spectra_All = torch.tensor(np.array(fh[self.tag]['spectra'][:]))
            spectra_IDLip = torch.tensor(np.array(fh[self.tag]['LipidID_spectra'][:]))
            
            if i == 0:
                self.spectra_Lip = spectra_Lip
                self.spectra_All = spectra_All
                self.spectra_IDLip = spectra_IDLip
            else:
                self.spectra_Lip = torch.cat((self.spectra_Lip, spectra_Lip), dim=0)
                self.spectra_All = torch.cat((self.spectra_All, spectra_All), dim=0)
                self.spectra_IDLip = torch.cat((self.spectra_IDLip, spectra_IDLip), dim=0)

        
        spectra_energy = torch.sqrt(torch.sum(np.abs(self.spectra_All-self.spectra_IDLip)**2, dim=1))[:,None]
        self.spectra_All /= spectra_energy
        self.spectra_IDLip /= spectra_energy
        self.spectra_Lip /= spectra_energy

        self.s = self.spectra_All.shape


    def __len__(self):
        return self.s[0]
    
    def __getitem__(self, index):

        in1 = self.spectra_All[index]
        in2 = self.spectra_IDLip[index]
        out = self.spectra_Lip[index]

        phase = torch.exp(2*np.pi*1j*torch.rand(1))
        scale = torch.rand(1)+0.5
        in1, in2, out = in1*phase*scale, in2*phase*scale, out*phase*scale
        
        in1 = torch.stack((torch.real(in1), torch.imag(in1)), dim=0)
        in2 = torch.stack((torch.real(in2), torch.imag(in2)), dim=0)
        out = torch.stack((torch.real(out), torch.imag(out)), dim=0)

        return in1,in2,out

