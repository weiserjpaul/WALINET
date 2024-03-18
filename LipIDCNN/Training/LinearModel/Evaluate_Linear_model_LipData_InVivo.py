#!/usr/bin/python3

import os
import argparse
import numpy as np
import tools
import sys
import time
import matplotlib.pyplot as plt
import h5py
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from functools import partial

def main():

#Evaluate one model on a test set.
#Need to be modifiy depend of your input/output model/Data (2D/3D)
#Robuste code

    
    parser = argparse.ArgumentParser(description='Generate dataset.')
    parser.add_argument('-i1', '--dataLip', dest='filenameAll',required=True, help='Input Data with Lip (DataName.h5)')
    parser.add_argument('-i2', '--dataLipRm', dest='filenameLipRM',required=True, help='Input Data with Lip Removed by Op (DataName.h5)')
    parser.add_argument('-m', '--intput', dest='model_pathname',required=True,default='def.h5', help='Model to test (DataName.h5)')
    parser.add_argument('--nBatch', dest='NBatch',default=1024, help='Size of Mini-Batch (2⁵...2⁸)')
    
    args = parser.parse_args()
    NBatch = args.NBatch
    filenameAll = args.filenameAll
    filenameLipRM = args.filenameLipRM	
    model_pathname = args.model_pathname


    #To run only on the CPU , Use when you would train to model in same time
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   
    
    from keras.models import model_from_json
    from keras.models import load_model


    #__________________________________________________________________________________________________________________________________________-
    #Load the model
    #__________________________________________________________________________________________________________________________________________-
    
    # load json and create model
    json_file = open("%s/model.json" %(model_pathname), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s/weights.h5" %(model_pathname))
    print("Loaded model from disk")
 
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


    DataAll, FsD , NptD, N1D, N2D, NMRfreqD = tools.load_LipidStackdata(filenameAll)
    
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


    DataLipRM, FsD , NptD, N1D, N2D, NMRfreqD = tools.load_LipidStackdata(filenameLipRM)
    
    if (0 & ( (FsD != Fs) | (N1D != N1) | (N2D != N2) |(NptD != Npt) |(NMRfreqD != NMRFreq) )) :
        print("FsD= ", FsD)
        print("Fs= ", Fs)
        print("Npt= ", Npt)
        print("shape(DataLipRM)[1]= ", np.shape(DataLipRM)[1])
        print('Model and dataset do not fit! (Line 104 in Evaluate script)')
        return 0

    
#__________________________________________________________________________________________________________________________________________-
#Evaluate the model
#__________________________________________________________________________________________________________________________________________-

    DataLipRM = DataLipRM[0:NBatch,:]
    DataAll = DataAll[0:NBatch,:]

    spectra_energy = np.sqrt(np.sum(np.abs(DataAll)**2, axis=1))[:,None]

   # print('np.shape(spectra_energy): {}'.format(np.shape(spectra_energy)))
    DataAll /= spectra_energy
    In1 = np.stack((np.real(DataAll), np.imag(DataAll)), axis=-1)

    DataLipRM /= spectra_energy
    In2 = np.stack((np.real(DataLipRM), np.imag(DataLipRM)), axis=-1)   
 
    print("DataLip shape: ", DataAll.shape)
    print("DataLipRM shape: ", DataLipRM.shape)
    #spectra_stack = spectra_stack[..., None]
    ## pad wrap to get Nbex,114,3

    #print('In.shape = ', In.shape) # (NbEx, 104,2)
    ## "wrap" padding to add 8 up/down in the spectral dimension and replicate the imaginary part
    #In1 = np.pad(In1, ((0,0),(8, 8),(0,0)), 'edge') 
    #In1 = np.pad(In1, ((0,0),(0, 0),(1,0)), 'wrap')
    #In1 = np.expand_dims(In1, axis = -1)

    #In2 = np.pad(In2, ((0,0),(8, 8),(0,0)), 'edge')
    #In2 = np.pad(In2, ((0,0),(0, 0),(1,0)), 'wrap')
    #In2 = np.expand_dims(In2, axis = -1)
    In = np.concatenate((In1,In2),axis=1)

    print('In.shape = ', In.shape) # (NbEx, 104,2)
    In = np.expand_dims(In, axis = -1)    

    Predicted_LipRM = loaded_model.predict(In)
    Predicted_LipRM = Predicted_LipRM.reshape((In1.shape[0], In1.shape[1],In1.shape[2]))
    Predicted_LipRM = np.squeeze(Predicted_LipRM[:,:,0]+ 1j*Predicted_LipRM[:,:,1])
    #Predicted_LipRM=np.squeeze(Predicted_LipRM[:,8:(np.shape(Predicted_LipRM)[1]-8),0]+ 1j*Predicted_LipRM[:,8:(np.shape(Predicted_LipRM)[1]-8),1]) 
    print("Predicted_LipRM shape: ", Predicted_LipRM.shape)
    Predicted_LipRM *= spectra_energy
    DataLipRM *= spectra_energy
    DataAll *= spectra_energy

#__________________________________________________________________________________________________________________________________________-
#plot the result
#__________________________________________________________________________________________________________________________________________-
    
    global fig,ax, k0
    fig, ax = plt.subplots(4)
    k0 = (int)(0.5*np.shape(DataLipRM)[0])
    def draw_spectra():
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()
        ax[3].clear()
        ax[0].plot(np.real(DataLipRM[k0, :].T))
        ax[1].plot(np.imag(DataLipRM[k0, :].T)) 
        ax[0].plot(np.real(Predicted_LipRM[k0, :].T))
        ax[1].plot(np.imag(Predicted_LipRM[k0, :].T))
        ax[2].plot(np.real(DataLipRM[k0, :].T-Predicted_LipRM[k0, :].T))
        ax[3].plot(np.imag(DataLipRM[k0, :].T-Predicted_LipRM[k0, :].T))  
        ax[0].set_title('Real')
        ax[1].set_title('Imag')
        ax[2].set_title('Real Diff')
        ax[3].set_title('Imag Diff')
        #ax[2].get_xaxis().set_visible(False)
        #ax[2].get_yaxis().set_visible(False)
        #ax[2].axis('off')
    draw_spectra()
    def onclick(fig, event):
        global k0
        if event.key == 'up':
            k0 = (k0 + 1) % (DataLipRM.shape[0])
        elif event.key == 'down':
            k0 = (k0 - 1) % (DataLipRM.shape[0])

        draw_spectra()
        plt.draw()

    fig.canvas.mpl_connect('key_press_event', partial(onclick, fig))

   
    plt.show()
    
if __name__ == '__main__':
    main()

