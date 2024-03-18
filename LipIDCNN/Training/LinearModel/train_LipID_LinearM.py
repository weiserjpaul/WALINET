#!/usr/bin/python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tools
import pickle
import os
import argparse
import h5py
import sys
import time
import json
import importlib

#Change the keras backend
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Conv2D, MaxPooling1D, PReLU, Dropout, AveragePooling1D, Reshape
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import LSTM,TimeDistributed, Activation, ConvLSTM2D, GRU, Input, Embedding, RNN, BatchNormalization
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model
from keras.callbacks import CSVLogger

import visual_callbacks

#from functools import partial

def main():

#Training of one model
#Need to be modifiy dependof your input/output data/model (2D/3D)
#Robust code

    parser = argparse.ArgumentParser(description='Generate dataset.')
    parser.add_argument('-o', '--output', dest='model_Dir',required=True)
    parser.add_argument('-i', '--intput', dest='filename',required=True)


    parser.add_argument('--Nbatch', dest='Nbatch',default = 256)
    parser.add_argument('--Nepochs', dest='Nepochs',default = 30)
    parser.add_argument('--NMLPlayer', dest='nMLPlayer',default = 3)
    parser.add_argument('--Nneuron', dest='nNeuron',default = 450)
    parser.add_argument('--Nfilters', dest='nFilters',default = 8) #12)
    parser.add_argument('--GPUpartition', dest='GPUpartition',default = 0)
    parser.add_argument('--DropOut', dest='DropOut',default = 0.01)
    parser.add_argument('--lr', dest='lr',default = 0.01)
    parser.add_argument('--optimizer', dest='optimizer',default = 'adam')
    parser.add_argument('--Tlayer', dest='tLayer',default = 'PReLU')
    parser.add_argument('--regularizer', dest='regularizer',default = False)

    args = parser.parse_args()
    model_Dir = args.model_Dir
    filename = args.filename
    
    #Constante
    Nbatch = int(args.Nbatch)
    Nepochs = int(args.Nepochs)
    nMLPlayer = int(args.nMLPlayer)
    nNeuron = int(args.nNeuron)
    nFilters = int(args.nFilters)
    GPUpartition = float(args.GPUpartition)
    DropOut = float(args.DropOut)
    lr = float(args.lr)
    optimizer = args.optimizer
    tLayer = args.tLayer
    regularizer = args.regularizer

    #NMRFreq = 123.2625 * 1e6 
    #WINDOW_START = 4.2      #ppm # 40
    #WINDOW_END = 1.0        #ppm # 200

    pathname = filename #"../Datasets/%s"%(filename)
    model_pathname = model_Dir #"./Model/%s"%(model_Dir)

    if not os.path.exists(model_pathname):
        os.makedirs(model_pathname)

#__________________________________________________________________________________________________________________________________________-
#Load Tensorflow, Keras and parametrize the GPU
#__________________________________________________________________________________________________________________________________________-

    # Partition gpu memory
    import tensorflow as tf
    if True :
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        
        if GPUpartition == 0 :
            config.gpu_options.allow_growth = True                                      #To allow GPU to manadge alone its memory
        else:
            config.gpu_options.per_process_gpu_memory_fraction = GPUpartition           #To fraction manually the GPU memory
        
        check = True
        while check:
            #Control if tensor can allocate request memory
            try:
                set_session(tf.Session(config=config))
                check = False
            except:
                print("Error during memory allocation")
                time.sleep(5)
                check = True
    else :
        #To run only on the CPU , Use when you would train to model in same time
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   
    
    import keras
    from keras import backend as K
    from keras.models import model_from_json
    from keras.utils.vis_utils import plot_model

#______________________________________________________________________________________________-
#Load training set
#__________________________________________________________________________________________________________________________________________-
    #added SNR here but not in tools.py
    data_train = tools.load_data3(pathname,
                                load=('train/spectra',
                                	'train/Lipid_BL_Wat_spectra','train/LipidID_spectra','train/Metab_spectra',
                                    'train/amplitudes',
                                    'train/spectra/max',
                                    'train/spectra/energy',
                                    'train/index','train/SNR'
                                     ))
    # meta-data of each array is loaded automatically with
    Fs=data_train['train/spectra/Fs'] 
    Npt=data_train['train/spectra/Npt']
    NMRFreq=data_train['train/spectra/NMRFreq']
    WINDOW_START=data_train['train/spectra/WINDOW_START']
    WINDOW_END=data_train['train/spectra/WINDOW_END']
    N1=data_train['train/spectra/N1']
    N2=data_train['train/spectra/N2']

    spectra_clean_max = data_train['train/spectra/max']
    spectra_clean_energy = data_train['train/spectra/energy'] 
    #BL_spectra = data_train['train/BL_spectra']
    index = data_train['train/index']
    names = data_train['train/names']
    names = names + ['SNR']
    print(names)
    ##added params
    snr = data_train['train/SNR']

    #spectra     = data_train['train/spectra'][:, N1:N2]
    spectra_Lip = data_train['train/Lipid_BL_Wat_spectra']
    spectra_All = data_train['train/spectra'][:]
    spectra_IDLip = data_train['train/LipidID_spectra'][:]
    spectra_Metab = data_train['train/Metab_spectra'][:]

    #amplitudes    = data_train['train/amplitudes'][:]

    # lets shuffle the data (numpy does not use a fixed seed, you have to explicitly ask for it with
    np.random.seed(0) # for example
    perm = np.random.permutation(spectra_All.shape[0])
    spectra_All = spectra_All [perm,:]
    spectra_Lip = spectra_Lip [perm,:]
    spectra_IDLip = spectra_IDLip [perm,:]
    spectra_Metab = spectra_Metab[perm,:]
    spectra_IDMetab = spectra_All - spectra_IDLip
    #Normalize Training Input data
     

    #print('np.shape(spectra): {}'.format(np.shape(spectra_IDMetab)))
    #spectra_energy = np.sqrt(np.sum(np.abs(spectra_All-spectra_IDLip)**2, axis=1))[:,None] #Energy of the metabolite signal
    spectra_energy_All = np.sqrt(np.sum(np.abs(spectra_All)**2, axis=1))[:,None] #Energy of the metabolite signal
    spectra_energy_MetabID = np.sqrt(np.sum(np.abs(spectra_IDMetab)**2, axis=1))[:,None] #Energy of the metabolite signal
    spectra_energy = spectra_energy_MetabID
    #spectra_energy[ range(0, spectra_energy_All.shape[0], 2)] =  spectra_energy_All[ range(0, spectra_energy_All.shape[0], 2)]
    #spectra_energy[ range(1, spectra_energy_All.shape[0], 2)] =  spectra_energy_MetabID[ range(1, spectra_energy_All.shape[0], 2)]
    #print('np.shape(spectra_energy): {}'.format(np.shape(spectra_energy)))
    spectra_All /= spectra_energy
    spectra_All_stack = np.stack((np.real(spectra_All), np.imag(spectra_All)), axis=-1)
    
    spectra_IDMetab /= spectra_energy
    spectra_IDMetab_stack = np.stack((np.real(spectra_IDMetab), np.imag(spectra_IDMetab)), axis=-1)
    
    spectra_Metab /= spectra_energy
    spectra_Metab_stack  = np.stack((np.real(spectra_Metab), np.imag(spectra_Metab)), axis=-1)

    #spectra_IDLip /= spectra_energy
    #spectra_IDLip_stack = np.stack((np.real(spectra_IDLip), np.imag(spectra_IDLip)), axis=-1)
    
    #spectra_Lip /= spectra_energy
    #spectra_Lip_stack  = np.stack((np.real(spectra_Lip), np.imag(spectra_Lip)), axis=-1)

    ## Tau11 / NAA12 / Glc13 / Ins14 / Gln15 / PCh16 

    ##Names with indexation
    ##GPC+PCh 0 / Glu+Gln 1 / NAA+NAAG 2 / PCr+Cr 3 / Scy 4 / Glc 5 / Lax 6 / Ins 7 / Ala 8 / Tau 9 / GSH 10
    ## Asp 11 / GABA 12
    #In1 = spectra_All_stack
    #In2 = spectra_IDLip_stack
    #In2 = spectra_IDMetab_stack

    #print('In.shape = ', In.shape) # (NbEx, 104,2)
    ## "wrap" padding to add 8 up/down in the spectral dimension and replicate the imaginary part
    #In1 = np.pad(In1, ((0,0),(8, 8),(0,0)), 'edge') 
    #In1 = np.pad(In1, ((0,0),(0, 0),(1,0)), 'wrap')
    #In1 = np.expand_dims(In1, axis = -1)

    #In2 = np.pad(In2, ((0,0),(8, 8),(0,0)), 'edge')
    #In2 = np.pad(In2, ((0,0),(0, 0),(1,0)), 'wrap')
    #In2 = np.expand_dims(In2, axis = -1)
    In = np.concatenate((spectra_All_stack,spectra_IDMetab_stack),axis=2)
    print('In.shape = ', In.shape) # (NbEx, 104,2)
    In = np.expand_dims(In, axis = -1)
    Out = spectra_Metab_stack
    #Out = spectra_Lip_stack
    #Out = np.pad(Out, ((0,0),(8, 8),(0,0)), 'edge')
    #Out = np.pad(Out, ((0,0),(0, 0),(1,0)), 'wrap') # No need to pad the complex dimension
    Out = Out.reshape((Out.shape[0], Out.shape[1]*Out.shape[2]))
    #Out = np.expand_dims(Out, axis = -1)
    print('Out.shape = ', Out.shape) # (NbEx, 104,2)
    
#__________________________________________________________________________________________________________________________________________-
#Train the model
#__________________________________________________________________________________________________________________________________________-
    #import architecture_model
    import modulable_CNN2

    #print('In.shape expanded = ', In.shape) #previous result before padding: (NbEx, 104,2,1)
 
    model = modulable_CNN2.LinearModel(input_shape= (In.shape[1],In.shape[2],1,), output_shape= (Out.shape[1]),nFilters=nFilters, nNeuron=nNeuron, nMLPlayer=nMLPlayer, drop=DropOut)
    #model = modulable_CNN2.DenseModel(input_shape= (In.shape[1],In.shape[2],1,), output_shape= (Out.shape[1]),nFilters=nFilters, nNeuron=248, nMLPlayer=2, drop=DropOut)

    V_splits = 0.01
    #Tloss = 'logcosh' 
    Tloss = 'mse' 
    
    if optimizer == 'adam':
        opimizer = keras.optimizers.adam(lr=lr)
    elif optimizer == 'RMSprop':
        opimizer = keras.optimizers.RMSprop(lr=lr)
    
    #model.compile(loss='mean_squared_error',
    #              optimizer= optimizer, metrics=['mse']) 
    model.compile(loss=Tloss,
                  optimizer= optimizer, metrics=['mse']) 

    model.summary()
                   
       
    csv_logger = CSVLogger("%s/epoch_log.csv"%(model_pathname), separator=',', append=False)
    
# to use in callbacks below...
    class LossHistory(keras.callbacks.Callback):
         def on_train_begin(self, logs={}):
             self.losses = []
 
         def on_batch_end(self, batch, logs={}):
             self.losses.append(logs.get('loss'))
    
    callbacks = [
    #plotter,
                   keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=20, verbose=1), csv_logger,
                   keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, verbose=1)
                   ]
    
  
    t = time.time()
    train_history = model.fit( In ,Out,
                              epochs=Nepochs,
                              validation_split=V_splits,
                              batch_size=Nbatch, 
                              callbacks= callbacks,
                              verbose=1
                              )
                              
                              
    elapsed = np.round(time.time() - t, decimals=4)             #Time to train the model
    print("Time to train : %s [s]"%(elapsed))
#              check = True
    train_history = train_history.history


 
#__________________________________________________________________________________________________________________________________________-
#Save model
#__________________________________________________________________________________________________________________________________________-
    #Parameters&Hyperparameters&Values

    Dico_value = {  "Model name :" : model_pathname,
                    "Data loaded :": filename,
                    "Time to train [s] :" : elapsed,
                    "Mini-batch size :" : Nbatch,
                    "Number of Epochs :" : Nepochs,
                    "Train set size :" : In.shape[0] * (1-V_splits),
                    "Valid set size :" : In.shape[0] * V_splits,
                    "Optimizer :" : opimizer,
                    "Loss function :" : Tloss,
                    #"Time Early Stop patience :" : StopPatience,
                    "Number of parameters :" : model.count_params(),
                    "Last Valid loss value :" : train_history['val_loss'][-1],
                    "Last Train loss value :" : train_history['loss'][-1],
                    "Valid losses values :" : train_history['val_loss'][:],
                    "Train losses values :" : train_history['loss'][:],
                    #"R2 :" : r2,
                    "Output names :" : names,
                    "DropOut Value :" : DropOut,
                    "Learning rate Value :" : lr
    }
    #Value in text file
    with open("%s/Values.txt" %(model_pathname), "w") as log:
        log.write("INTERESTING VALUES \n")
        for keys in Dico_value.keys():
            log.write('%s %s \n' % ( keys , Dico_value.get(keys) ))
            
#    with open("%s/model_log.csv"%(model_pathname),'w', newline = '') as csvFile:
#        wr = csv.writer(csvFile, quoting=csv.QUOTE_ALL)
#        wr.writerow(RESULTS)
    
        
    #Save Weights
    model.save_weights("%s/weights.h5" %(model_pathname), overwrite=True)

    with h5py.File("%s/params.h5" %(model_pathname), 'w', libver='earliest') as f:
        train = f.create_group('Model')
        #train.create_dataset('y_span', data=y_span)
        #train.create_dataset('y_min', data=y_min)
        train.create_dataset('Fs', data=Fs) 
        train.create_dataset('Npt', data=Npt) 
        train.create_dataset('spectra_energy', data=spectra_energy) 
        train.create_dataset('NMRFreq', data=NMRFreq) 
        train.create_dataset('WINDOW_START', data=WINDOW_START) 
        train.create_dataset('WINDOW_END', data=WINDOW_END) 
        train.create_dataset('N1', data=N1) 
        train.create_dataset('N2', data=N2) 

    #model
    model_json = model.to_json()
    with open("%s/model.json" %(model_pathname), "w") as json_file :
        json_file.write(model_json)
    #Image of layer
    plot_model(model, to_file= "%s/Structur.png" %(model_pathname), 
                show_shapes=True, show_layer_names=True)
    #Training History
        
    print("Saved model to disk in folder %s" %(model_pathname))


######## plotting
    print('Plotting')

    out_predict = model.predict(In)
    out_predictTemp = np.squeeze(out_predict[:,8:(np.shape(out_predict)[1]-8),:])
    out_predict=np.squeeze(out_predict[:,8:(np.shape(out_predict)[1]-8),0]+ 1j*out_predict[:,8:(np.shape(out_predict)[1]-8),1]) 
    


    global fig, ax
    
    fig, ax = plt.subplots(nrows=2)
    plt.tight_layout()
    plt.suptitle(model_Dir, fontsize=12)
    ax = ax.flatten()
    print('out_predictTemp shape: ',out_predictTemp.shape)
    print('spectra_Lip_stack shape: ',spectra_Lip_stack.shape)
    def draw_spectra(example):
        for i in range(2):
            ax[i].clear()
            #ax[i].plot((Out_predict2[example , :,i]))
            ax[i].plot((spectra_All_stack[example , :,i] - out_predictTemp[example , :,i]))
            #ax[i].plot((spectra_stack_crop_abs[example , :,i]), '--')
            ax[i].plot((spectra_All_stack[example , :,i] - spectra_Lip_stack[example , :,i]), '--')
            ax[i].plot((spectra_All_stack[example , :,i] - spectra_IDLip_stack[example , :,i]), '.-')
            #ax[i].plot((BL_spectra_stack_crop_abs[example , :]), '.-')
            #ax[i].plot((spectra_BL_stack[example, :,i]), '--')
            ax[i].legend(['Predicted', 'Ground Truth', 'Lip.Supp.Op Output'])
            ax[i].grid(True)
    #n0 = 0
    nameFig = "{}/resultSpectrum{}.png"
    for j in range(20):
        draw_spectra(j)
        plt.savefig(nameFig.format(model_pathname,j), dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)
        #n0 += 1
    plt.close(fig)
 
if __name__ == '__main__':
    main()
