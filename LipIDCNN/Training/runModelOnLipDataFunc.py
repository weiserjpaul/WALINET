#!/usr/bin/python3
import sys
import os
import argparse
import numpy as np
from make_CNN_StackData_rrrt.LipIDCNN.Training import tools
import sys
import time
import matplotlib.pyplot as plt
import h5py
from functools import partial

from keras.models import model_from_json
from keras.models import load_model



def run(image_rrrf, skMask, headmask, NMRfreq, MinPPM_Lip, MaxPPM_Lip, ppm):

    #################################
    ### Lipid Projection Operator ###
    #################################

    Data_rrrf = image_rrrf
    s = Data_rrrf.shape
    beta=1E+3 * .3 #0.938

    skMask = np.array(fh[h5tag]["SkMask"])
    skMask = np.transpose(skMask, axes=(1,2,0))

    Data_rf = np.reshape(Data_rrrf, (s[0]*s[1]*s[2],s[3]))
    lipid_mask = np.reshape(skMask, (s[0]*s[1]*s[2]))

    lipid_rf = Data_rf[lipid_mask>0,:]


    LipidRem_Operator_ff = np.linalg.inv(np.eye(s[-1]) + beta * np.matmul(np.conj(lipid_rf).T, lipid_rf))
    LipidProj_cff = np.eye(s[4])-LipidRem_Operator_ff

    ############################
    ### Lipid Suppression NN ###
    ############################

    MaxPPM_pt_In=np.argmin(np.abs(MaxPPM_Lip - ppm))
    MinPPM_pt_In=np.argmin(np.abs(MinPPM_Lip - ppm))
    MaxPPM_pt_In=MaxPPM_pt_In - np.mod((MaxPPM_pt_In-MinPPM_pt_In+1),16)+16
    
    N1=MinPPM_pt_In
    N2=MaxPPM_pt_In

    LipidStackHead = image_rrrf[headmask>0,:]
    DataAll = (LipidStackHead[:,N1:N2+1], s[4], N1, N2+1, NMRfreq)

    LipidProjStackHead = np.matmul(image_rrrf[headmask>0,:], LipidProj_ff)
    DataLipID = (LipidProjStackHead[:,N1:N2+1], s[4], N1, N2+1, NMRfreq)
    
    model_pathname = '/autofs/space/somes_002/users/pweiser/lipidSuppressionCNN/models/EXP_1/'
    (Predicted_LipID, Fs, Npt, N1, N2, NMRFreq) = runModelOnLipData(model_pathname, DataAll, DataLipID)

    Data_Lipid_rrrf = np.zeros(headmask.shape+(s[4],), dtype=np.complex64)
    Data_Lipid_rrrf[headmask>0,:N2] = Predicted_LipID

    Data_LipidRemoved_rrrf = image_rrrf - Data_Lipid_rrrf
    
    return Data_LipidRemoved_rrrf



def runModelOnLipData(model_pathname, DataAll, DataLipID):

    #__________________________________________________________________________________________________________________________________________-
    #Load the model
    #__________________________________________________________________________________________________________________________________________-

    print("Loading model from disk")
    # load json and create model
    json_file = open("%s/model.json" %(model_pathname), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s/weights.h5" %(model_pathname))
    
 
    # evaluate loaded model on test data
    loaded_model.compile(loss='mse', optimizer='adam', metrics=[tools.R2])
    

    params = tools.load_params("%s/params.h5" %(model_pathname),
                                load=('Model/Fs','Model/Npt','Model/spectra_energy','Model/NMRFreq',
                                    'Model/WINDOW_START','Model/WINDOW_END','Model/N1','Model/N2'))

    # meta-data of each array is loaded automatically with
    Fs=params['Model/Fs'] 
    Npt=params['Model/Npt'] 
    spectra_energy=params['Model/spectra_energy'] 
    NMRFreq=params['Model/NMRFreq'] 
    WINDOW_START=params['Model/WINDOW_START'] 
    WINDOW_END=params['Model/WINDOW_END']
    N1=params['Model/N1'] 
    N2=params['Model/N2']  


    #__________________________________________________________________________________________________________________________________________-
    #Load test set 1
    #__________________________________________________________________________________________________________________________________________-


    DataAll, FsD , NptD, N1D, N2D, NMRfreqD = DataAll
    
    if (0 & ( (FsD != Fs) | (N1D != N1) | (N2D != N2) |(NptD != Npt) |(NMRfreqD != NMRFreq))) :
        print("FsD= ", FsD)
        print("Fs= ", Fs)
        print("Npt= ", Npt)
        print("shape(DataAll)[1]= ", np.shape(DataAll)[1])
        print('Model and dataset do not fit! (Line 87 in Evaluate script)')
        return 0



    #__________________________________________________________________________________________________________________________________________-
    #Load test set 2
    #__________________________________________________________________________________________________________________________________________-


    DataLipID, FsD , NptD, N1D, N2D, NMRfreqD = DataLipID
    
    if (0 & ( (FsD != Fs) | (N1D != N1) | (N2D != N2) |(NptD != Npt) |(NMRfreqD != NMRFreq) )) :
        print("FsD= ", FsD)
        print("Fs= ", Fs)
        print("Npt= ", Npt)
        print("shape(DataLipID)[1]= ", np.shape(DataLipID)[1])
        print('Model and dataset do not fit! (Line 104 in Evaluate script)')
        return 0


    #__________________________________________________________________________________________________________________________________________-
    #Evaluate the model
    #__________________________________________________________________________________________________________________________________________-
    print('Scaling and formatting the input data.')

    spectra_energy = np.sqrt(np.sum(np.abs(DataAll - DataLipID)**2, axis=1))[:,None]

    spectra_energy[np.where(spectra_energy==0)]=1

    DataAll /= spectra_energy
    In1 = np.stack((np.real(DataAll), np.imag(DataAll)), axis=-1)

    DataLipID /= spectra_energy
    In2 = np.stack((np.real(DataLipID), np.imag(DataLipID)), axis=-1)   
 
    print("DataLip shape: ", DataAll.shape)
    print("DataLipID shape: ", DataLipID.shape)
   
    ## "wrap" padding to add 8 up/down in the spectral dimension and replicate the imaginary part
    In1 = np.pad(In1, ((0,0),(8, 8),(0,0)), 'edge') 
    In1 = np.pad(In1, ((0,0),(0, 0),(1,0)), 'wrap')
    In1 = np.expand_dims(In1, axis = -1)

    In2 = np.pad(In2, ((0,0),(8, 8),(0,0)), 'edge')
    In2 = np.pad(In2, ((0,0),(0, 0),(1,0)), 'wrap')
    In2 = np.expand_dims(In2, axis = -1)

    print('The model is running.')
    Predicted_LipID = loaded_model.predict([In1, In2])
    Predicted_LipID = np.squeeze(Predicted_LipID[:,8:(np.shape(Predicted_LipID)[1]-8),0]+ 1j*Predicted_LipID[:,8:(np.shape(Predicted_LipID)[1]-8),1]) 
    #print("Predicted_LipID shape: ", Predicted_LipID.shape)
    Predicted_LipID *= spectra_energy

    return (Predicted_LipID, Fs, Npt, N1, N2, NMRFreq)