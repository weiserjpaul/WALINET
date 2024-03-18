#!/usr/bin/python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from make_CNN_StackData_rrrt.LipIDCNN.Training import tools
#import tools
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
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Conv2D, MaxPooling1D, PReLU, Dropout, AveragePooling1D, Reshape
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import LSTM,TimeDistributed, Activation, ConvLSTM2D, GRU, Input, Embedding, RNN, BatchNormalization
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import Sequence
from keras.callbacks import CSVLogger

#import visual_callbacks

#from functools import partial

def main(raw_args=None):

#Training of one model
#Need to be modifiy dependof your input/output data/model (2D/3D)
#Robust code

    parser = argparse.ArgumentParser(description='Generate dataset.')
    parser.add_argument('-o', '--output', dest='model_Dir',required=True)
    parser.add_argument('-i', '--intput', dest='filename',required=True)


    parser.add_argument('--Nbatch', dest='Nbatch',default = 512)
    parser.add_argument('--Nepochs', dest='Nepochs',default = 30)
    parser.add_argument('--nMLPlayer', dest='nMLPlayer',default = 5)
    parser.add_argument('--nNeuron', dest='nNeuron',default = 50)
    parser.add_argument('--nFilters', dest='nFilters',default = 12) #12)
    parser.add_argument('--GPUpartition', dest='GPUpartition',default = 0)
    parser.add_argument('--dropOut', dest='dropOut',default = 0.10)
    parser.add_argument('--lr', dest='lr',default = 0.01)#0.01)
    parser.add_argument('--optimizer', dest='optimizer',default = 'adam')
    parser.add_argument('--tLayer', dest='tLayer',default = 'PReLU')
    parser.add_argument('--regularizer', dest='regularizer',default = False)

    args = parser.parse_args(raw_args)
    model_Dir = args.model_Dir
    filename = args.filename
    
    #Constante
    Nbatch = int(args.Nbatch)
    Nepochs = int(args.Nepochs)
    nMLPlayer = int(args.nMLPlayer)
    nNeuron = int(args.nNeuron)
    nFilters = int(args.nFilters)
    GPUpartition = float(args.GPUpartition)
    dropOut = float(args.dropOut)
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
    if 0 :
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
        # 0 to use the first GPU, 1 the second, -1 To run only on the CPU , Use when you would train to model in same time
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'   
    select_gpu(0)
    import keras
    from keras import backend as K
    from keras.models import model_from_json
    from keras.utils.vis_utils import plot_model

#______________________________________________________________________________________________-
#Load training set
#__________________________________________________________________________________________________________________________________________-
    print('Loading training dataset...')
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
    spectra_Lip = data_train['train/Lipid_BL_Wat_spectra'][:]
    spectra_All = data_train['train/spectra'][:]
    spectra_IDLip = data_train['train/LipidID_spectra'][:]
    #spectra_Metab = data_train['train/Metab_spectra'][:]

    #amplitudes    = data_train['train/amplitudes'][:]

    # lets shuffle the data (numpy does not use a fixed seed, you have to explicitly ask for it with
    np.random.seed(0) # for example
    perm = np.random.permutation(spectra_All.shape[0])
    spectra_All = spectra_All[perm,:]
    spectra_Lip = spectra_Lip[perm,:]
    spectra_IDLip = spectra_IDLip[perm,:]
    #spectra_Metab = spectra_Metab[perm,:]
    #spectra_IDMetab = spectra_All - spectra_IDLip
    #Normalize Training Input data
     

    #print('np.shape(spectra): {}'.format(np.shape(spectra_IDMetab)))
    spectra_energy = np.sqrt(np.sum(np.abs(spectra_All-spectra_IDLip)**2, axis=1))[:,None] #Energy of the metabolite signal
    #spectra_energy = np.sqrt(np.sum(np.abs(spectra_All)**2, axis=1))[:,None] #Energy of the metabolite signal

    #print('np.shape(spectra_energy): {}'.format(np.shape(spectra_energy)))
    spectra_All /= spectra_energy
    spectra_All_stack = np.stack((np.real(spectra_All), np.imag(spectra_All)), axis=-1)
    del spectra_All
    #spectra_IDMetab /= spectra_energy
    #spectra_IDMetab_stack = np.stack((np.real(spectra_IDMetab), np.imag(spectra_IDMetab)), axis=-1)
    
    #spectra_Metab /= spectra_energy
    #spectra_Metab_stack  = np.stack((np.real(spectra_Metab), np.imag(spectra_Metab)), axis=-1)

    spectra_IDLip /= spectra_energy
    spectra_IDLip_stack = np.stack((np.real(spectra_IDLip), np.imag(spectra_IDLip)), axis=-1)
    del spectra_IDLip

    spectra_Lip /= spectra_energy
    spectra_Lip_stack  = np.stack((np.real(spectra_Lip), np.imag(spectra_Lip)), axis=-1)
    del spectra_Lip

    ## Tau11 / NAA12 / Glc13 / Ins14 / Gln15 / PCh16 

    ##Names with indexation
    ##GPC+PCh 0 / Glu+Gln 1 / NAA+NAAG 2 / PCr+Cr 3 / Scy 4 / Glc 5 / Lax 6 / Ins 7 / Ala 8 / Tau 9 / GSH 10
    ## Asp 11 / GABA 12
    In1 = spectra_All_stack
    In2 = spectra_IDLip_stack
    #In2 = spectra_IDMetab_stack

    print('In1.shape = ', In1.shape) # (NbEx, 104,2)
    print('In2.shape = ', In2.shape) # (NbEx, 104,2)
    ## "wrap" padding to add 8 up/down in the spectral dimension and replicate the imaginary part
    In1 = np.pad(In1, ((0,0),(8, 8),(0,0)), 'edge') 
    In1 = np.pad(In1, ((0,0),(0, 0),(1,0)), 'wrap')
    In1 = np.expand_dims(In1, axis = -1)
    #In1 = tf.convert_to_tensor(In1)

    In2 = np.pad(In2, ((0,0),(8, 8),(0,0)), 'edge')
    In2 = np.pad(In2, ((0,0),(0, 0),(1,0)), 'wrap')
    In2 = np.expand_dims(In2, axis = -1)
    #In2 = tf.convert_to_tensor(In2)

    print('In1.shape = ', In1.shape) # (NbEx, 104,2)
    print('In2.shape = ', In2.shape) # (NbEx, 104,2)

    #Out = spectra_Metab_stack
    Out = spectra_Lip_stack
    Out = np.pad(Out, ((0,0),(8, 8),(0,0)), 'edge')
    #Out = np.pad(Out, ((0,0),(0, 0),(1,0)), 'wrap') # No need to pad the complex dimension
    Out = np.expand_dims(Out, axis = -1)
    #Out = tf.convert_to_tensor(Out)
    print('Out.shape = ', Out.shape) # (NbEx, 104,2)
    
#__________________________________________________________________________________________________________________________________________-
#Train the model
#__________________________________________________________________________________________________________________________________________-
    print('Loading the model...')
    #import architecture_model
    #import modulable_CNN2
    from make_CNN_StackData_rrrt.LipIDCNN.Training import modulable_CNN2

    #print('In.shape expanded = ', In.shape) #previous result before padding: (NbEx, 104,2,1)
 
    model = modulable_CNN2.model_UNet_Lipid_sup_dist_Correction(inputLip_shape= (In1.shape[1],In1.shape[2],1,),inputCorr_shape=(In2.shape[1],In2.shape[2],1,), tLayer=tLayer, regularizer=regularizer,
                                    nFilters=nFilters, nNeuron=nNeuron, nMLPlayer=nMLPlayer, drop=dropOut)

    if True:
        print("Loading model from disk")
        #load_model_pathname = '/autofs/space/somes_002/users/pweiser/lipidSuppressionCNN/models/EXP_19_19sub_AllAcc_Wat'
        load_model_pathname = '/autofs/space/somes_002/users/pweiser/lipidSuppressionCNN/models/EXP_31_19subv2'
        print("Load from " + load_model_pathname)
        # load json and create model
        json_file = open("%s/model.json" %(load_model_pathname), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("%s/weights.h5" %(load_model_pathname))

    
    V_splits = 0.01
    #Tloss = 'logcosh' 
    Tloss = 'mse' 
    #Tloss = dual_domain_wrapper()

    if optimizer == 'adam':
        #opimizer = keras.optimizers.adam_v2.Adam(learning_rate=lr)
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(lr=lr)
    
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
                   keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=20, verbose=1), 
                   csv_logger,
                   keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, verbose=1),
                   keras.callbacks.ModelCheckpoint(model_pathname+'/checkpoint', save_best_only=True, monitor='val_loss', mode='min')
                   ]
    
  
    t = time.time()
    print('Training the model...')
    '''
    train_history = model.fit( [In1, In2] ,Out,
                              epochs=Nepochs,
                              validation_split=V_splits,
                              batch_size=Nbatch, 
                              callbacks= callbacks,
                              verbose=1
                              )
    '''
    samples = In1.shape[0]
    v_index = int((1-V_splits)*samples)
    train_gen = DataGenerator(In1[:v_index], In2[:v_index], Out[:v_index], Nbatch)
    test_gen = DataGenerator(In1[v_index:], In2[v_index:], Out[v_index:], Nbatch)

    train_history = model.fit(train_gen,
                            epochs=Nepochs,
                            validation_data=test_gen,
                            callbacks= callbacks,
                            verbose=1)
                              
                              
    elapsed = np.round(time.time() - t, decimals=4)             #Time to train the model
    print("Time to train : %s [s]"%(elapsed))
#              check = True
    train_history = train_history.history

    #out_predict = model.predict([In1, In2])
    #out_predictTemp = np.squeeze(out_predict[:,8:(np.shape(out_predict)[1]-8),:])
    #out_predict=np.squeeze(out_predict[:,8:(np.shape(out_predict)[1]-8),0]+ 1j*out_predict[:,8:(np.shape(out_predict)[1]-8),1]) 

 
#__________________________________________________________________________________________________________________________________________-
#Save model
#__________________________________________________________________________________________________________________________________________-
    #Parameters&Hyperparameters&Values

    Dico_value = {  "Model name :" : model_pathname,
                    "Data loaded :": filename,
                    "Time to train [s] :" : elapsed,
                    "Mini-batch size :" : Nbatch,
                    "Number of Epochs :" : Nepochs,
                    "Train set size :" : In1.shape[0] * (1-V_splits),
                    "Valid set size :" : In1.shape[0] * V_splits,
                    "Optimizer :" : optimizer,
                    #"Loss function :" : Tloss,
                    #"Time Early Stop patience :" : StopPatience,
                    "Number of parameters :" : model.count_params(),
                    "Last Valid loss value :" : train_history['val_loss'][-1],
                    "Last Train loss value :" : train_history['loss'][-1],
                    "Valid losses values :" : train_history['val_loss'][:],
                    "Train losses values :" : train_history['loss'][:],
                    #"R2 :" : r2,
                    "Output names :" : names,
                    "DropOut Value :" : dropOut,
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
    #plot_model(model, to_file= "%s/Structur.png" %(model_pathname), 
    #            show_shapes=True, show_layer_names=True)
    #Training History
        
    print("Saved model to disk in folder %s" %(model_pathname))

"""
######## plotting
    print('Plotting')
    


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
        plt.savefig(nameFig.format(model_pathname,j), dpi=None, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
        #n0 += 1
    plt.close(fig)
"""

def select_gpu(gpu_id=-1, max_usage=.8):  # max 2 gpu only
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id != -1 else '0,1'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    max_memory = 30000#11534  # MB got from: grep -i --color memory /var/log/Xorg.0.log
    for gpu in gpus:
        print('GPU FOUND:', gpu)
        tf.config.experimental.set_memory_growth(gpu, True)  # FIXME true
        tf.config.experimental.set_virtual_device_configuration(gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_memory * max_usage)])
    print('RUNNING ON GPU #{}'.format(gpu_id))

class DataGenerator(Sequence):
    def __init__(self, in1_set, in2_set, out_set, batch_size):
        self.in1, self.in2, self.out = in1_set, in2_set, out_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.in1) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_in1 = self.in1[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_in2 = self.in2[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_out = self.out[idx * self.batch_size:(idx + 1) * self.batch_size]
        return [batch_in1, batch_in2], batch_out



if __name__ == '__main__':
    main()
