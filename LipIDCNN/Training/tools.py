import matplotlib.pyplot as plt
import os, io, pickle
import h5py
import keras.backend as K
import keras
import pandas as pd
import numpy as np
from numpy import array as a
import csv
import glob
import sys
import time
from sklearn.metrics import r2_score
from functools import partial
import tensorflow as tf

#Robust script.

#__________________________________________________________________________________________________________________________________________-
#METRICS
#__________________________________________________________________________________________________________________________________________-
def rscore(x, y):
    return K.mean((x - K.mean(x, axis=0)) * (y - K.mean(y, axis=0)), axis=0) / (K.std(x, axis=0) * K.std(y, axis=0))
def R2(y_true, y_pred):
    return K.mean(rscore(y_true, y_pred) ** 2)
def mseRe(y_true, y_pred):
    return 1/tf.cast(tf.size(y_true), dtype=tf.float32) * 1/4 * K.sum( (y_true - y_pred)**2 )
def pente(x, y):
    s = tf.cast(tf.size(x), dtype=tf.float32)
    return ( K.sum(x)*K.sum(y) - s*K.sum(x*y) ) / ( K.sum(x)**2 - s*K.sum(x**2) )

def np_rscore(x, y):
    return np.mean((x - np.mean(x, axis=0)) * (y - np.mean(y, axis=0)), axis=0) / (np.std(x, axis=0) * np.std(y, axis=0))
def np_R2(x,y):
    return np.mean( np_rscore(x,y)**2 )
#__________________________________________________________________________________________________________________________________________-
#__________________________________________________________________________________________________________________________________________-

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes 

def csv_read(filename):
    values = np.array([])
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        n_row = 0
        row_size = 0
        for row in reader:
            if row[0].startswith('#'):
                continue
            n_row = n_row + 1
            row_size = len(row)

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        values = np.zeros((n_row, row_size))
        n_row = 0
        for row in reader:
            if row[0].startswith('#'):
                continue
            values[n_row, :] = np.array([float(v)
                                         for v in row]).reshape((1, 3))
            n_row = n_row + 1

    return values

## added BL_spectra in loaded data
def load_data3(filename,
               load=('train/spectra', 'train/amplitudes', 'train/spectra/max', 'train/spectra/energy', 'train/BL_spectra'),
               join=(('Cr', 'PCr'), ('NAA', 'NAAG'), ('GPC', 'PCh'), ('Glu', 'Gln'))):
    if not os.path.exists(filename):
        print('ERROR: File {} does not exist.'.format(filename))
        return {}

    import h5py
    f = h5py.File(filename, 'r')

    loaded_data = {}

    for key in load:
        splits = key.split('/')
        index = dict([(k.decode('utf8'), int(v)) for (k, v) in f[splits[0]+'/'+'index']])
        names = [None] * len(index)
        for k, v in index.items():
            names[v] = k
        names_key = splits[0]+'/names'
        loaded_data[names_key] = names
        to_join = []
        for (name1, name2) in join:
            mode1 = index[name1]
            mode2 = index[name2]
            to_join.append((mode1, mode2))

        to_join = sorted([(min(m), max(m)) for m in to_join])

        for m in to_join:
            names[m[0]] = names[m[0]] + '+' + names[m[1]]

        for m in to_join:
            names.remove(names[m[1]])

        # this is an array load
        if len(splits) == 2:
            if splits[1] == 'spectra':
                # TODO Should we do spectra pre-processing here on in the script using this??
                loaded_data[key] = f[key]
                for akey, avalue in f[key].attrs.items():
                    loaded_data[key + '/' + akey] = avalue

            elif splits[1] == 'BL_spectra':
                loaded_data[key] = f[key]
                for akey, avalue in f[key].attrs.items():
                    loaded_data[key + '/' + akey] = avalue
                    
            elif splits[1] == 'spectra_clean':
                loaded_data[key] = f[key]
                for akey, avalue in f[key].attrs.items():
                    loaded_data[key + '/' + akey] = avalue

            elif splits[1] == 'amplitudes':
                data = f[key][:]
                for m in to_join:
                    data[:, m[0]] += data[:, m[1]]

                for m in reversed(sorted(to_join, key=lambda mm: mm[1])):
                    data = np.delete(data, m[1], 1)
                loaded_data[key] = data
                for akey, avalue in f[key].attrs.items():
                    loaded_data[key + '/' + akey] = avalue

            elif splits[1] == 'metab_spectra':
                data = f[key][:]
                for m in to_join:
                    data[m[0], :] += data[m[1], :]

                for m in reversed(sorted(to_join, key=lambda mm: mm[1])):
                    data = np.delete(data, m[1], 0)
                loaded_data[key] = data
                for akey, avalue in f[key].attrs.items():
                    loaded_data[key + '/' + akey] = avalue
            else:
                loaded_data[key] = f[key][:]

        # this an array attribute load
        if len(splits) == 3:
            loaded_data[key] = f[splits[0] + '/' + splits[1]].attrs[splits[2]]

    return loaded_data

## added BL_spectra in loaded data
def load_params(filename,
               load=('Model/y_span','Model/y_min','Model/Fs','Model/Npt','Model/spectra_energy','Model/NMRFreq','Model/WINDOW_START','Model/WINDOW_END','Model/N1','Model/N2')):
    if not os.path.exists(filename):
        print('ERROR: File {} does not exist.'.format(filename))
        return {}

    import h5py
    f = h5py.File(filename, 'r')

    loaded_data = {}

    for key in load:
        print('Loading {}'.format(key))
        loaded_data[key] = f[key][()]

    return loaded_data

def load_InVivodata(filename):
    if not os.path.exists(filename):
        print('ERROR: File {} does not exist.'.format(filename))
        return {}
    arrays = {}
    import h5py
    import numpy as np
    #import pickle
    with h5py.File(filename, 'r', libver='earliest') as f:
        for k, v in f.items():
             arrays[k] = np.array(v)

    BrainMask=arrays['BrainMask']
    MrProt=arrays['MrProt']
    if np.ndim(arrays['realData']) == 4:
        Data_trrr=arrays['realData']+ 1j* arrays['imagData']
        return Data_trrr, BrainMask, MrProt
    elif np.ndim(arrays['realData']) == 3:
        Data_trr=arrays['realData']+ 1j* arrays['imagData']
        return Data_trr, BrainMask, MrProt
    else:
        print('ERROR: Data Format Unknown in file: ',filename)
        return {}
#_______________________________________________________________________________________________
#_______________________________________________________________________________________________
#_______________________________________________________________________________________________


def load_LipidStackdata(filename):
    if not os.path.exists(filename):
        print('ERROR: File {} does not exist.'.format(filename))
        return {}
    arrays = {}
    import h5py
    import numpy as np
    #import pickle
    with h5py.File(filename, 'r', libver='earliest') as f:
        for k, v in f.items():
             arrays[k] = np.array(v)

    samplerate=arrays['samplerate']
    Npt=int(arrays['Npt'])
    N1=int(arrays['N1'])
    N2=int(arrays['N2'])
    NMRfreq=arrays['NMRfreq']
    
    print(arrays.keys())
    if 'realData' in arrays.keys():
        Data_rf=np.transpose(arrays['realData']+ 1j* arrays['imagData'],(1,0))
    else:
        Data_rf=arrays['Lipid_Stack_File']
    #Data_rf=arrays['realData']+ 1j* arrays['imagData']
    return Data_rf,  samplerate, Npt, N1, N2,NMRfreq

#_______________________________________________________________________________________________
#_______________________________________________________________________________________________
#_______________________________________________________________________________________________


def MakeTrainingSpectralData(NbEx=10000, MaxSNR=5.0, MinSNR=1.0, MaxFreq_Shift=20.0,
                             MaxPeak_Width=30.0, MinPeak_Width=10.0, NbBL=10,MinBLWidth=80, MaxAcquDelay=2E-3, Fs = 4000 ,N = 1056,modes_filter='MetabModes/*Exact_Modes.txt',wstart=4.2,wend=1.0,NMRFreq=123.2625 * 1e6, LipProjFile='', verbose=True ):
    """ Generates stuff

    Args:
        NbEx(int)               : Number of realizations
        MinSNR(float)           : Minimal Signal to Noise Ratio
        MaxSNR(float)           : Maximal Signal to Noise Ratio
        MaxFreq_Shift(float)    : Maximal Frequency Shift
        MaxPeak_Width(float)    : Maximal Peak Width
    """
    import h5py
    import numpy as np
    from numpy import pi as PI

  #  WINDOW_START = 4.2      #now in function arguments
  #  WINDOW_END = 1.0        #now in function arguments
    N1 = int(np.around(((4.7-wstart) * 1e-6 * NMRFreq)* N / Fs))
    N2 = int(np.around(((4.7-wend) * 1e-6 * NMRFreq)* N / Fs))
    if verbose:
        print('Generating data ... ')

    TimeSerie = np.zeros(( N), dtype=np.complex64)
    TimeSerieClean = np.zeros(( N), dtype=np.complex64)
    TimeSerieLipRM = np.zeros(( N), dtype=np.complex64)
    BLTimeSerie = np.zeros(( N), dtype=np.complex64)

    Spectra = np.zeros((NbEx, (N2-N1)), dtype=np.complex64)
    SpectraClean = np.zeros((NbEx, (N2-N1)), dtype=np.complex64)
    LipRM_Spectra = np.zeros((NbEx, (N2-N1)), dtype=np.complex64)
    BLSpectra = np.zeros((NbEx, (N2-N1)), dtype=np.complex64)

    #comboLorGauss = np.zeros((NbEx,N), dtype=np.complex64)
    #print('shape of TimeSerie ={}.'.format(TimeSerie[N1:N2].shape))
    #print('N1={}, N2={}, number of points in reduced data ={}.'.format(N1,N2,(N2-N1+1)))
    Time = np.linspace(0, N - 1, N) / Fs 
    Frequencies = np.linspace(0, Fs, N)
    list_file = glob.glob(modes_filter)
    NbM = len(list_file)

    metabo_modes = [None] * NbM  # empty list of size NbM
    index = {}  # mapping of metabolite name to index

    mean_std_csv = pd.read_csv(
        'MetabModes/Metab_Mean_STD.txt', header=None).values


    for i, v in enumerate(mean_std_csv[:, 0].astype(str)):
        index[ bytes(v.strip(), 'utf8') ] = i
    if verbose:
        print(index)
        print(mean_std_csv)

    mean_std = mean_std_csv[:, 1:].astype(np.float32)

    for i, filename in enumerate(list_file):
        metabo_mode = pd.read_csv(
            filename, header=None, skiprows=[0]).values
        a = filename.find('3T_') + 3
        b = filename.find('_Exact')
        name = bytes(filename[a:b].strip(), 'utf8')
        metabo_modes[index[name]] = metabo_mode

    from numpy.random import rand, randn, seed
    seed()

    # Amplitude = rand(NbEx, NbM)
    # numpy behavior: NxM * 1xM -> multiply all rows by the 1xM
    # Amplitude = mean_std[:, 0] + (rand(NbEx, NbM)-0.5)*mean_std[:, 1]

    #Amplitude = (mean_std[:, 0] + mean_std[:, 1] / 2) * rand(NbEx, NbM)
    #use mean as std and multiply by .08 to avoid too many outliers
    Amplitude = np.absolute((mean_std[:, 0])*0.8 * randn(NbEx, NbM) + mean_std[:, 0])

    FreqShift = (rand(NbEx, 1)*2 - 1) * MaxFreq_Shift
    PeakWidth = MinPeak_Width + rand(NbEx, 1) * (MaxPeak_Width - MinPeak_Width)
    ponder_peaks = rand(NbEx, 1)
    PeakWidth_Gau = np.multiply(ponder_peaks, PeakWidth)
    PeakWidth_Lor = np.multiply(1-ponder_peaks, PeakWidth)
    #PeakWidth_Gau = 0*(MinPeak_Width + rand(NbEx, 1) * (MaxPeak_Width - MinPeak_Width))
    PhShift = rand(NbEx, 1) * 2 * PI
    #AcquDelay =  (rand(NbEx, 1)-0.5)*2 * MaxAcquDelay
    AcquDelay =  (rand(NbEx, 1)) * MaxAcquDelay


    SNR = MinSNR + rand(NbEx, 1) * (MaxSNR - MinSNR)
    
    if LipProjFile:
        print('Loading Lipid Projection operator ({} ).'.format(LipProjFile))
        arrays = {}

        with h5py.File(LipProjFile, 'r', libver='earliest') as f:
            for k, v in f.items():
                arrays[k] = np.array(v)
        LipidProj_ffc= np.transpose( arrays['realLipidProj']+ 1j* arrays['imagLipidProj'],(2,1,0))
        CoilLP=np.round((rand(NbEx)*LipidProj_ffc.shape[2]-0.5)*0.999)
    
    TempMetabData = np.zeros((len(metabo_modes), N), dtype=np.complex64)
    for f, mode in enumerate(metabo_modes):
        Freq = ((4.7-mode[:, 0]) * 1e-6 * NMRFreq)[...,None]
        for Nuc in range(len(Freq)):
            TempMetabData[f, :] += mode[Nuc, 1][...,None] * np.exp(1j * mode[Nuc, 2][...,None]) * np.exp(2 * PI * 1j * (Time)  * Freq[Nuc])  

    print('np.shape(TempMetabData): {}'.format(np.shape(TempMetabData)))     
    
    for ex in range(NbEx):
        if verbose:
            if np.mod(ex, int(NbEx / 100)) == 0:
                print('{}% '.format(int(ex * 100 / NbEx)), end='', flush=True)    
        
        #Weighted sum of Lor and Gauss: we add the lorentzian and gaussian time serie (each weighted by 1-n or n) in order to make a mixture of both   
        #physical meaning is less clear compared to the Voigt function            
        #comboLorGauss[ex,:] = (ponder_peaks[ex]* (np.exp(1j * PhShift[ex] +(Time) * (- PeakWidth_Lor[ex] )))) + ((1-ponder_peaks[ex]) * np.exp ( (1j * PhShift[ex]) -((Time**2) * (PeakWidth_Gau[ex]**2))))
        TimeSerieClean=0*TimeSerieClean
        for f, _ in enumerate(metabo_modes): 
            ##  Lorentzian spreading of metabolite modes peaks
            #TimeSerieClean[ex, :] += Amplitude[ex, f] * TempMetabData[f, :]* np.exp(1j * PhShift[ex] + (Time) * (- PeakWidth_Lor[ex] ))
            ##  Gaussian spreading of metabolite modes peaks
            #TimeSerieClean[ex, :] += Amplitude[ex, f] * TempMetabData[f, :]* np.exp(1j * PhShift[ex] + (-((Time**2) * (PeakWidth_Gau[ex]**2))))
            ##  Gaussian and Lorentzian weighted sum metabolite modes peaks spreading
            ## uncomment the line "comboLorGauss[..." 
            #TimeSerieClean[ex, :] += Amplitude[ex, f] * TempMetabData[f, :]* comboLorGauss[ex,:]
            # Voigt profile 
            TimeSerieClean[:] += Amplitude[ex, f] * TempMetabData[f, :]* np.exp(1j * PhShift[ex])  

        BLTimeSerie=0*BLTimeSerie
        for ga in range(NbBL):
            AmpGauss=5*Amplitude[ex,:].max()*rand(1);
            TimeGauss=rand(1)/MinBLWidth
            PPMG=4.7*rand(1)
            FreqGauss=- ( PPMG - 4.7 ) * 1e-6 * NMRFreq
            PhGaus= 2 * PI * rand(1)
            BLTimeSerie[:] += AmpGauss*np.exp(1j*PhGaus + (Time) * 2 * PI * 1j* FreqGauss -Time**2 /(TimeGauss**2))
        
        SpectrumTemp = np.fft.fft(TimeSerieClean[ :],axis=0) 
        TimeSerieClean[:] = np.fft.ifft(SpectrumTemp* np.exp(2 * PI *1j * -AcquDelay[ex]*Frequencies),axis=0) * np.exp( (Time* 1j * 2 * PI * FreqShift[ex] ) + (- (np.square(Time)) * (np.square(PeakWidth_Gau[ex]))) + ((np.absolute(Time)) * (- PeakWidth_Lor[ex]) ))
        SpectrumTemp = np.fft.fft(TimeSerieClean[:],axis=0) 

        NCRand=(randn(N) + 1j * randn(N))
        TimeSerie[:] = TimeSerieClean[:] + BLTimeSerie[:] + np.fft.ifft(SpectrumTemp[N1:N2].std()/0.65 / SNR[ex] * NCRand,axis=0)

        if LipProjFile:
            SpectrumTempLipidProj=SpectrumTemp-np.dot(SpectrumTemp,LipidProj_ffc[:,:,int(CoilLP[ex])])
            TimeSerieLipRM[:] = np.fft.ifft(SpectrumTempLipidProj,axis=0)
            TimeSerieLipRM[:] += BLTimeSerie[:] + np.fft.ifft(SpectrumTemp[N1:N2].std()/0.65 / SNR[ex] * NCRand,axis=0)    
        else:
            TimeSerieLipRM[:]=TimeSerie[:]
        
        Spectra[ex,:] = np.fft.fft(TimeSerie, axis=0)[N1:N2]
        SpectraClean[ex,:] = np.fft.fft(TimeSerieClean, axis=0)[N1:N2]
        LipRM_Spectra[ex,:] = np.fft.fft(TimeSerieLipRM, axis=0)[N1:N2]
        BLSpectra[ex,:] = np.fft.fft(BLTimeSerie, axis=0)[N1:N2]

    MetabSpectra = np.fft.fft(TempMetabData, axis=1)[:,N1:N2]

    if verbose:
        print('100%', flush=True)
    return Spectra, Amplitude, MetabSpectra, index, SpectraClean, BLSpectra,LipRM_Spectra, FreqShift, PhShift, PeakWidth_Lor,PeakWidth_Gau, SNR, AcquDelay



#_______________________________________________________________________________________________
#_______________________________________________________________________________________________
#_______________________________________________________________________________________________

def load_timeSerie(filename,
               load=('train/timeserie', 'train/amplitudes'),
               join=(('Cr', 'PCr'), ('NAA', 'NAAG'), ('GPC', 'PCh'), ('Glu', 'Gln'))):
    if not os.path.exists(filename):
        print('ERROR: File {} does not exist.'.format(filename))
        return {}

    import h5py
    f = h5py.File(filename, 'r')

    loaded_data = {}

    for key in load:
        splits = key.split('/')
        index = dict([(k.decode('utf8'), int(v)) for (k, v) in f[splits[0]+'/'+'index']])
        names = [None] * len(index)
        for k, v in index.items():
            names[v] = k
        names_key = splits[0]+'/names'
        loaded_data[names_key] = names
        to_join = []
        for (name1, name2) in join:
            mode1 = index[name1]
            mode2 = index[name2]
            to_join.append((mode1, mode2))

        to_join = sorted([(min(m), max(m)) for m in to_join])

        for m in to_join:
            names[m[0]] = names[m[0]] + '+' + names[m[1]]

        for m in to_join:
            names.remove(names[m[1]])

        # this is an array load
        if len(splits) == 2:
            if splits[1] == 'spectra':
                # TODO Should we do spectra pre-processing here on in the script using this??
                loaded_data[key] = f[key]
                for akey, avalue in f[key].attrs.items():
                    loaded_data[key + '/' + akey] = avalue

            elif splits[1] == 'timeSerie':
                loaded_data[key] = f[key]
                for akey, avalue in f[key].attrs.items():
                    loaded_data[key + '/' + akey] = avalue

            elif splits[1] == 'spectra_clean':
                loaded_data[key] = f[key]
                for akey, avalue in f[key].attrs.items():
                    loaded_data[key + '/' + akey] = avalue

            elif splits[1] == 'amplitudes':
                data = f[key][:]
                for m in to_join:
                    data[:, m[0]] += data[:, m[1]]

                for m in reversed(sorted(to_join, key=lambda mm: mm[1])):
                    data = np.delete(data, m[1], 1)
                loaded_data[key] = data
                for akey, avalue in f[key].attrs.items():
                    loaded_data[key + '/' + akey] = avalue

            elif splits[1] == 'metab_spectra':
                data = f[key][:]
                for m in to_join:
                    data[m[0], :] += data[m[1], :]

                for m in reversed(sorted(to_join, key=lambda mm: mm[1])):
                    data = np.delete(data, m[1], 0)
                loaded_data[key] = data
                for akey, avalue in f[key].attrs.items():
                    loaded_data[key + '/' + akey] = avalue
            else:
                loaded_data[key] = f[key][:]

        # this an array attribute load
        if len(splits) == 3:
            loaded_data[key] = f[splits[0] + '/' + splits[1]].attrs[splits[2]]

    return loaded_data

#_______________________________________________________________________________________________
#_______________________________________________________________________________________________
#Plot histogram Data

def histo_data (Data):
    n, bins, patches = plt.hist(x=Data, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('DataOUt Histogram')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()


#_______________________________________________________________________________________________
#_______________________________________________________________________________________________
#_______________________________________________________________________________________________


def load_LipidOpdata(filename):
    if not os.path.exists(filename):
        print('ERROR: File {} does not exist.'.format(filename))
        return {}
    arrays = {}
    import h5py
    import numpy as np
    #import pickle
    with h5py.File(filename, 'r', libver='earliest') as f:
        for k, v in f.items():
             arrays[k] = np.array(v)

    samplerate=arrays['samplerate']
    Npt=int(arrays['Npt'])
    N1=int(arrays['N1'])
    N2=int(arrays['N2'])
    NMRfreq=arrays['NMRfreq']
    
    if 'realLipidProj' in arrays.keys():
        Data_cff=np.transpose(arrays['realLipidProj']+ 1j* arrays['imagLipidProj'],(2,1,0))
    else:
        Data_cff=arrays['LipidProj']
    #Data_rf=arrays['realData']+ 1j* arrays['imagData']
    return Data_cff,  samplerate, Npt, N1, N2,NMRfreq


