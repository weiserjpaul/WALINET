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
#Need to be modifiy depend of your input/outpuHarmonyOS,t model/Data (2D/3D)
#Robuste code

    
    parser = argparse.ArgumentParser(description='Generate dataset.')
    parser.add_argument('-i', '--data', dest='data',required=True, help='Input Data (DataName.h5)')
    parser.add_argument('-m', '--intput', dest='model_pathname',required=True,default='def.h5', help='Model to test (DataName.h5)')
    parser.add_argument('--nBatch', dest='NBatch',default=256, help='Size of Mini-Batch (2⁵...2⁸)')
    
    args = parser.parse_args()
    NBatch = args.NBatch
    pathname = args.data
    model_pathname = args.model_pathname


    #To run only on the CPU , Use when you would train to model in same time
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   
    
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
    N1=params['Model/N1'] test/
    N2=params['Model/N2']  


#__________________________________________________________________________________________________________________________________________-
#Load training set
#__________________________________________________________________________________________________________________________________________-
    #added SNR here but not in tools.py
    data_test = tools.load_data3(pathname,
                                load=('test/spectra',
                                	'test/Lipid_BL_Wat_spectra','test/LipidID_spectra','test/Metab_spectra',
                                    'test/amplitudes',
                                    'test/spectra/max',
                                    'test/spectra/energy',
                                    'test/index','test/SNR'
                                     ))

    # meta-data of each array is loaded automatically with
    Fs=data_test['test/spectra/Fs'] 
    Npt=data_test['test/spectra/Npt']
    NMRFreq=data_test['test/spectra/NMRFreq']
    WINDOW_START=data_test['test/spectra/WINDOW_START']
    WINDOW_END=data_test['test/spectra/WINDOW_END']
    N1=data_test['test/spectra/N1']
    N2=data_test['test/spectra/N2']

    spectra_clean_max = data_test['test/spectra/max']
    spectra_clean_energy = data_test['test/spectra/energy'] 
    #BL_spectra = data_test['test/BL_spectra']
    index = data_test['test/index']
    names = data_test['test/names']
    names = names + ['SNR']
    print(names)
    ##added params
    snr = data_test['test/SNR']

    spectra_Lip = data_test['test/Lipid_BL_Wat_spectra']
    spectra_All = data_test['test/spectra'][:]
    spectra_IDLip = data_test['test/LipidID_spectra'][:]
    spectra_Metab = data_test['test/Metab_spectra'][:]
    spectra_IDMetab = spectra_All - spectra_IDLip

    # lets shuffle the data (numpy does not use a fixed seed, you have to explicitly ask for it with
    np.random.seed(0) # for example
    #perm = np.random.permutation(spectra.shape[0])
    #spectra = spectra [perm,:]
    #spectra_Lip = spectra_Lip [perm,:]

    #Normalize Training Input data
     

    spectra_energy = np.sqrt(np.sum(np.abs(spectra_All-spectra_IDLip)**2, axis=1))[:,None]
    #print('np.shape(spectra_energy): {}'.format(np.shape(spectra_energy)))
    spectra_All /= spectra_energy
    spectra_All_stack = np.stack((np.real(spectra_All), np.imag(spectra_All)), axis=-1)
    
    spectra_IDLip /= spectra_energy
    spectra_IDLip_stack = np.stack((np.real(spectra_IDLip), np.imag(spectra_IDLip)), axis=-1)

    spectra_Lip /= spectra_energy
    spectra_Lip_stack  = np.stack((np.real(spectra_Lip), np.imag(spectra_Lip)), axis=-1)

    ## Tau11 / NAA12 / Glc13 / Ins14 / Gln15 / PCh16 

    ##Names with indexation
    ##GPC+PCh 0 / Glu+Gln 1 / NAA+NAAG 2 / PCr+Cr 3 / Scy 4 / Glc 5 / Lax 6 / Ins 7 / Ala 8 / Tau 9 / GSH 10
    ## Asp 11 / GABA 12



 
    
#__________________________________________________________________________________________________________________________________________-
#Evaluate the model
#__________________________________________________________________________________________________________________________________________-

    print("spectra_stack shape: ", spectra_All_stack.shape)

    #spectra_stack = spectra_stack[..., None]
    ## pad wrap to get Nbex,114,3

    In1 = spectra_All_stack
    In2 = spectra_IDLip_stack
    #print('In.shape = ', In.shape) # (NbEx, 104,2)
    ## "wrap" padding to add 8 up/down in the spectral dimension and replicate the imaginary part
    In1 = np.pad(In1, ((0,0),(8, 8),(0,0)), 'edge') 
    In1 = np.pad(In1, ((0,0),(0, 0),(1,0)), 'wrap')
    In1 = np.expand_dims(In1, axis = -1)

    In2 = np.pad(In2, ((0,0),(8, 8),(0,0)), 'edge')
    In2 = np.pad(In2, ((0,0),(0, 0),(1,0)), 'wrap')
    In2 = np.expand_dims(In2, axis = -1)


    PredictedLip = loaded_model.predict([In1, In2])
  test/
    PredictedLip=np.squeeze(PredictedLip[:,8:(np.shape(PredictedLip)[1]-8),0]+ 1j*PredictedLip[:,8:(np.shape(PredictedLip)[1]-8),1]) 

    print("PredictedLip shape: ", PredictedLip.shape)
    PredictedLip *= spectra_energy
    spectra_All *= spectra_energy
    spectra_IDLip *= spectra_energy
    spectra_Lip *= spectra_energy
    PredictedMetab = spectra_All - PredictedLip
    spectra_IDMetab = spectra_All - spectra_IDLip
    ActualMetab = spectra_All - spectra_Lip
#__________________________________________________________________________________________________________________________________________-
#save the result
#__________________________________________________________________________________________________________________________________________-   
    with h5py.File(filename, 'a', libver='earliest') as f:
        train = f.create_group('test')
        dset = train.create_dataset('spectra', data=spectra_All, fletcher32=True)
        dset.attrs['Fs'] = Fs
        dset.attrs['Npt'] = Npt
        dset.attrs['NMRFreq'] = NMRFreq
        dset.attrs['WINDOW_START'] = WINDOW_START
        dset.attrs['WINDOW_END'] = WINDOW_END
        dset.attrs['N1'] = N1
        dset.attrs['N2'] = N2

        train.create_dataset('Metab_spectra', data=ActualMetab, fletcher32=True)
        train.create_dataset('Lipid_spectra', data=spectra_Lip, fletcher32=True)
        train.create_dataset('LipidID_spectra', data=spectra_IDLip, fletcher32=True)
        
        train.create_dataset('Predicted_Metab_spectra', data=PredictedMetab, fletcher32=True)
        train.create_dataset('Predicted_Lipid_spectra', data=PredictedLip, fletcher32=True)

#__________________________________________________________________________________________________________________________________________-
#plot the result
#__________________________________________________________________________________________________________________________________________-
    
    global fig,ax, k0
    fig, ax = plt.subplots(5)
    k0 = 0;#(int)(0.5*np.shape(Data_rf)[0])
    MaxPt = np.shape(spectra_All)[1]-8-1
    def draw_spectra():
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()
        ax[3].clear()
        ax[4].clear()
        ax[0].plot(np.real(spectra_Lip[k0, 8:MaxPt].T),label='Actual Lipids')
        ax[1].plot(np.imag(spectra_Lip[k0, 8:MaxPt].T))
        #ax[0].plot(np.real(spectra_All[k0, 8:MaxPt].T),label='Original Spectrum ')
        #ax[1].plot(np.imag(spectra_All[k0, 8:MaxPt].T))
        ax[0].plot(np.real(spectra_IDLip[k0, 8:MaxPt].T),label='Lin.Op IDed Lipids')
        ax[1].plot(np.imag(spectra_IDLip[k0, 8:MaxPt].T)) 
        ax[0].plot(np.real(PredictedLip[k0, 8:MaxPt].T),label='AI IDed Lipids')
        ax[1].plot(np.imag(PredictedLip[k0, 8:MaxPt].T))



        ax[2].plot(np.real(ActualMetab[k0, 8:MaxPt].T),label='Actual Metabolites')
        ax[3].plot(np.imag(ActualMetab[k0, 8:MaxPt].T))  
        ax[4].plot(np.abs(ActualMetab[k0, 8:MaxPt].T)) 
        ax[2].plot(np.real(spectra_IDMetab[k0, 8:MaxPt].T),label='Lin.Op IDed Metabolites')
        ax[3].plot(np.imag(spectra_IDMetab[k0, 8:MaxPt].T))  
        ax[4].plot(np.abs(spectra_IDMetab[k0, 8:MaxPt].T)) 
        ax[2].plot(np.real(PredictedMetab[k0, 8:MaxPt].T),label='AI IDed Metabolites')
        ax[3].plot(np.imag(PredictedMetab[k0, 8:MaxPt].T))  
        ax[4].plot(np.abs(PredictedMetab[k0, 8:MaxPt].T)) 

        ax[0].set_title('Real')
        ax[1].set_title('Imag')
        ax[2].set_title('Real')
        ax[3].set_title('Imag')
        ax[4].set_title('Abs')
        ax[0].legend()
        ax[2].legend()
        #ax[2].get_xaxis().set_visible(False)
        #ax[2].get_yaxis().set_visible(False)
        #ax[2].axis('off')
    draw_spectra()
    def onclick(fig, event):
        global k0
        if event.key == 'up':
            k0 = (k0 + 1) % (spectra_All.shape[0])
        elif event.key == 'down':
            k0 = (k0 - 1) % (spectra_All.shape[0])

        draw_spectra()
        plt.draw()

    fig.canvas.mpl_connect('key_press_event', partial(onclick, fig))

   
    plt.show()
    
if __name__ == '__main__':
    main()

