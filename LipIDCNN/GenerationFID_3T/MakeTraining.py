
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


def MakeTrainingLipIDData(NbEx, MaxSNR, MinSNR, MaxFreq_Shift,MaxPeak_Width, MinPeak_Width, NbBL,MinBLWidth, MaxAcquDelay, MaxLipidScaling,MaxBLandWatScaling,LipStackFile='',LipOpFile='', verbose=True ):
    """ Generates stuff

    Args:
        NbEx(int)               : Number of realizations
        MinSNR(float)           : Minimal Signal to Noise Ratio
        MaxSNR(float)           : Maximal Signal to Noise Ratio
        MaxFreq_Shift(float)    : Maximal Frequency Shift
        MaxPeak_Width(float)    : Maximal Peak Width
    """
    import h5py
    import tools
    import numpy as np
    import os, inspect
    from numpy import pi as PI

  #  WINDOW_START = 4.2      #now in function arguments
  #  WINDOW_END = 1.0        #now in function arguments
    Lipid_rf, Fs, Npt, N1_lip, N2_lip, NMRFreq =tools.load_LipidStackdata(LipStackFile)
    print('shape of Lipid_rf ={}.'.format(Lipid_rf.shape))
    #if WaterStackFile:
    #    Water_rf, Fs, Npt, N1, N2, NMRFreq =tools.load_LipidStackdata(WaterStackFile)
    LipOp_cff, Fs, Npt, N1, N2,NMRFreq =tools.load_LipidOpdata(LipOpFile)
    LipOp_cff = np.array(LipOp_cff)
    print('shape of LipOp_cff ={}.'.format(LipOp_cff.shape))

    if verbose:
        print('Generating data ... ')

    #perm = np.random.permutation(np.shape(Lipid_rf)[0])
    #Lipid_rf = Lipid_rf[perm]
    if verbose:
        print('Generating data ... ')
    

    #print('np.shape(Lipid_rf): {}'.format(np.shape(Lipid_rf)))
    N = Npt #np.shape(Lipid_rf)[1]

    #N1 = int(np.around(((4.7-wstart) * 1e-6 * NMRFreq)* N / Fs))
    #N2 = int(np.around(((4.7-wend) * 1e-6 * NMRFreq)* N / Fs))

    TimeSerie = np.zeros(( N), dtype=np.complex64)
    TimeSerieClean = np.zeros(( N), dtype=np.complex64)
    LipTimeSerie = np.zeros(( N), dtype=np.complex64)
    BLTimeSerie = np.zeros(( N), dtype=np.complex64)
    WaterTimeSerie = np.zeros(( N), dtype=np.complex64)
    TempSpectrum = np.zeros(( N), dtype=np.complex64)

    AllInSpectra = np.zeros((NbEx, (N2-N1)), dtype=np.complex64)
    MetabSpectra = np.zeros((NbEx, (N2-N1)), dtype=np.complex64)
    LipidIDSpectra = np.zeros((NbEx, (N2-N1)), dtype=np.complex64)
    Lipid_BL_Wat_Spectra = np.zeros((NbEx, (N2-N1)), dtype=np.complex64)
    BLSpectra = np.zeros((NbEx, (N2-N1)), dtype=np.complex64)
    WaterSpectra = np.zeros((NbEx, (N2-N1)), dtype=np.complex64)
    Lipid_rf_Rand = np.zeros((NbEx, (N2_lip-N1_lip)), dtype=np.complex64)
    #comboLorGauss = np.zeros((NbEx,N), dtype=np.complex64)
    #print('shape of TimeSerie ={}.'.format(TimeSerie[N1:N2].shape))
    #print('N1={}, N2={}, number of points in reduced data ={}.'.format(N1,N2,(N2-N1+1)))
    Time = np.linspace(0, N - 1, N) / Fs 
    Frequencies = np.linspace(0, 1, N) * Fs

    ScriptPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    list_file = glob.glob(ScriptPath+'/MetabModes/*Exact_Modes.txt')
    NbM = len(list_file)

    metabo_modes = [None] * NbM  # empty list of size NbM
    index = {}  # mapping of metabolite name to index
    
    #print('ScriptPath: {}'.format(ScriptPath)) 
    mean_std_csv = pd.read_csv(ScriptPath+'/MetabModes/Metab_Mean_STD.txt', header=None).values

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

    Amplitude = np.ceil(mean_std[:, 0])*0.8 * randn(NbEx, NbM) + mean_std[:, 0]
    # putting negative values to 0
    Amplitude = Amplitude.clip(min=0)

    FreqShift = (rand(NbEx, 1)*2 - 1) * MaxFreq_Shift
    PeakWidth = MinPeak_Width + rand(NbEx, 1) * (MaxPeak_Width - MinPeak_Width)
    ponder_peaks = rand(NbEx, 1)
    PeakWidth_Gau = np.multiply(ponder_peaks, PeakWidth)
    PeakWidth_Lor = np.multiply(1-ponder_peaks, PeakWidth)
    #PeakWidth_Gau = 0*(MinPeak_Width + rand(NbEx, 1) * (MaxPeak_Width - MinPeak_Width))
    PhShift = rand(NbEx, 1) * 2 * PI
    AcquDelay =  (rand(NbEx, 1)-0.5)*2 * MaxAcquDelay
    #AcquDelay =  (rand(NbEx, 1)) * MaxAcquDelay
    LipidScaling =  1e-1*(10**(rand(NbEx, 1)*np.log10(1e1*MaxLipidScaling)))   
    #LipidScaling =  rand(NbEx, 1)*MaxLipidScaling;  # There is not much cases with Lipid<Metab in reality and in the Brain: Lipid ~100-1000 Metab signal
    WaterScaling=  0*1e-2*(10**(rand(NbEx, 1)*np.log10(1e2*MaxBLandWatScaling))); #1e-3*(10**(rand(NbEx, 1)*np.log10(1e3*MaxLipidScaling)))
    BLScaling =  1e-2*(10**(rand(NbEx, 1)*np.log10(1e2*MaxBLandWatScaling))); #rand(NbEx, 1)*LipidScaling + rand(NbEx, 1)*WaterScaling
    
    #print('LipidScaling: {}'.format(LipidScaling))     
    
    LipidPh = rand(NbEx, 1) * 2 * PI
    #LipPos=np.round((rand(NbEx)*np.shape(Lipid_rf)[0]-0.5)*0.9999999999)
    NbCombVox = 10
    for ex in range(NbEx):
        LipPos=np.round((rand(NbCombVox)*np.shape(Lipid_rf)[0]-0.5)*0.9999999999)
        #WaterPos=np.round((rand(1)*np.shape(Water_rf)[0]-0.5)*0.9999999999)
        #WAmp=(rand(1))
        LipAmp=(rand(NbCombVox));
        LipidPh= 2 * PI * rand(NbCombVox)
        LipAmp=LipAmp/np.sum(LipAmp)
        Lipid_rf_Rand[ex,:] = 0
        for LipV in range(NbCombVox):
            Lipid_rf_Rand[ex,:] += LipAmp[LipV]*np.exp(1j * LipidPh[LipV])*Lipid_rf[int(LipPos[LipV]),:]
    #if WaterStackFile:
    #    WaterPh = rand(NbEx, 1) * 2 * PI
    #    WaterScaling =  1e-3*(10**(rand(NbEx, 1)*np.log10(1e3*MaxLipidScaling)))
    #    for ex in range(NbEx):
    #        WaterPos=np.round((rand(1)*np.shape(Water_rf)[0]-0.5)*0.9999999999)
    #        Water_rf_Rand[ex,:] = WAmp*Water_rf[int(WaterPos),:]   

    SNR = MinSNR + rand(NbEx, 1) * (MaxSNR - MinSNR)
    
    
    TempMetabData = np.zeros((len(metabo_modes), N), dtype=np.complex64)
    for f, mode in enumerate(metabo_modes):
        Freq = ((4.7-mode[:, 0]) * 1e-6 * NMRFreq)[...,None]
        for Nuc in range(len(Freq)):
            TempMetabData[f, :] += mode[Nuc, 1][...,None] * np.exp(1j * mode[Nuc, 2][...,None]) * np.exp(2 * PI * 1j * (Time)  * Freq[Nuc])  

    #print('np.shape(TempMetabData): {}'.format(np.shape(TempMetabData)))     
    
    BatchSize = int(NbEx/200);
    print('Data generation Batch size: ({} ).'.format(BatchSize))
    LipidIDSpBatch = np.zeros((BatchSize, (N)), dtype=np.complex64)

    LipCoilInd = 0 
    BatchI = 0
    BlockI = 0

    for ex in range(NbEx):
        #print('ex: {}.'.format(ex))
        if verbose:
            if np.mod(ex, int(NbEx / 100)) == 0:
                print('{}% '.format(int(ex * 100 / NbEx)), end='', flush=True)    
        
        TimeSerieClean=0*TimeSerieClean
        for f, _ in enumerate(metabo_modes): 
            TimeSerieClean[:] += Amplitude[ex, f] * TempMetabData[f, :]* np.exp(1j * PhShift[ex])  

        BLTimeSerie=0*BLTimeSerie
        for ga in range(NbBL):
            AmpGauss=rand(1)
            TimeGauss=rand(1)/MinBLWidth
            TimeLor=rand(1)/MinBLWidth
            PPMG=6*rand(1)-1 # between 5 and -1 
            FreqGauss=- ( PPMG - 4.7 ) * 1e-6 * NMRFreq
            PhGaus= 2 * PI * rand(1)
            BLTimeSerie[:] += AmpGauss*np.exp(1j*PhGaus + (Time) * 2 * PI * 1j* FreqGauss -Time**2 /(TimeGauss**2) -Time /(TimeLor))
        
        TimeGauss=rand(1)/PeakWidth_Gau[ex]
        TimeLor=rand(1)/PeakWidth_Lor[ex]
        PhWater= 2 * PI * rand(1)
        FreqWater = (rand(1)*2-1)*20
        WaterTimeSerie[:] = np.exp(1j*PhWater + (Time) * 2 * PI * 1j* FreqWater -Time**2 /(TimeGauss**2) -Time /(TimeLor))

        SpectrumTemp = np.fft.fft(TimeSerieClean[ :],axis=0) 
        TimeSerieClean[:] = np.fft.ifft(SpectrumTemp* np.exp(2 * PI *1j * -AcquDelay[ex]*Frequencies),axis=0) * np.exp( (Time* 1j * 2 * PI * FreqShift[ex] ) + (- (np.square(Time)) * (np.square(PeakWidth_Gau[ex]))) + ((np.absolute(Time)) * (- PeakWidth_Lor[ex]) ) )
        SpectrumTemp = np.fft.fft(TimeSerieClean[:],axis=0) 

        NCRand=(randn(N) + 1j * randn(N))
        TimeSerie[:] = TimeSerieClean[:] + np.fft.ifft(SpectrumTemp[N1:N2].std()/0.65 / SNR[ex] * NCRand,axis=0)


        Metab_max = np.max(np.abs(np.fft.fft(TimeSerie, axis=0)[N1:N2]), axis=0)       
        #print('Metab_max: {}.'.format(Metab_max))
  
        if NbBL>0:
            BL_max =  np.max(np.abs(np.fft.fft(BLTimeSerie, axis=0)[N1:N2]), axis=0)
            BLSpectra[ex,:] = np.fft.fft(BLTimeSerie, axis=0)[N1:N2]*Metab_max/BL_max*BLScaling[ex]
        else:
            BLSpectra[ex,:] = 0

        Water_max =  np.max(np.abs(np.fft.fft(WaterTimeSerie, axis=0)[N1:N2]), axis=0)
        WaterSpectra[ex,:] = np.fft.fft(WaterTimeSerie, axis=0)[N1:N2]*Metab_max/Water_max*WaterScaling[ex]
   
        Lip_max =  np.max(np.abs(Lipid_rf_Rand[ex,N1:N2]), axis=0)
        #print('shape(Lipid_rf_Rand[ex,:]): {}.'.format(np.shape(Lipid_rf_Rand[ex,:])))
        #Lipid_BL_Wat_Spectra[ex, :] =   Metab_max/Lip_max*LipidScaling[ex] *Lipid_rf_Rand[ex,N1:N2] 
        LipTimeSerie[:] = np.fft.ifft(Lipid_rf_Rand[ex,:] , axis=0)
        TimeGauss=1.0/abs(PeakWidth_Gau[ex]*rand(1)) # remove a peak width <10 to compensate that Lip Spectra are already broad
        TimeLor=1.0/abs(PeakWidth_Lor[ex]*rand(1))
        LipTimeSerie *= np.exp(-Time**2 /(TimeGauss**2) -Time /(TimeLor))
        
        #Spectra with Full Lip, Water and metab
        Lipid_BL_Wat_Spectra[ex, :] =  BLSpectra[ex,:] + WaterSpectra[ex,:] +  Metab_max/Lip_max*LipidScaling[ex] *np.fft.fft(LipTimeSerie[:], axis=0)[N1:N2] 

        #Spectra with Removed Lip, Water and metab
        
        #LipidIDSpBatch[BatchI,:] =   Metab_max/Lip_max*LipidScaling[ex] *Lipid_rf_Rand[ex,:]
        LipidIDSpBatch[BatchI,:] =    Metab_max/Lip_max*LipidScaling[ex] *np.fft.fft(LipTimeSerie[:] , axis=0)      
        if NbBL>0:
            LipidIDSpBatch[BatchI,:] +=  np.fft.fft(BLTimeSerie, axis=0)*Metab_max/BL_max*BLScaling[ex] #Add baseline
        LipidIDSpBatch[BatchI,:] +=  np.fft.fft(WaterTimeSerie, axis=0)*Metab_max/Water_max*WaterScaling[ex] #Add Water
        LipidIDSpBatch[BatchI,:] += np.fft.fft(TimeSerie, axis=0) #Add Metabolites

        BatchI +=1      
        if BatchI==BatchSize:
            ExIndices = range((0 + BlockI*BatchSize),(BatchSize + BlockI*BatchSize))
            LipidIDSpectra[ExIndices,:] = np.dot(LipidIDSpBatch,(np.squeeze(LipOp_cff[LipCoilInd,:,:])))[:,N1:N2]

            BatchI = 0
            BlockI +=1
            LipCoilInd +=1
            if LipCoilInd == LipOp_cff.shape[0]:
                LipCoilInd = 0


        MetabSpectra[ex,:] = np.fft.fft(TimeSerie, axis=0)[N1:N2]

        AllInSpectra[ex, :] = MetabSpectra[ex, :] + Lipid_BL_Wat_Spectra[ex, :] 


        #SpectraClean[ex,:] = np.fft.fft(TimeSerieClean, axis=0)[N1:N2]

        #if WaterStackFile:
        #    Water_max =  np.max(np.abs(Water_rf_Rand[ex,:]), axis=0)
        #    Spectra[ex, :] +=  Metab_max/Lip_max*WaterScaling[ex]*np.exp(1j * WaterPh[ex]) *Water_rf_Rand[ex,:]
        

    SingleMetabSpectra = np.fft.fft(TempMetabData, axis=1)[:,N1:N2]

    if verbose:
        print('100%', flush=True)
    return AllInSpectra, Amplitude, MetabSpectra,Lipid_BL_Wat_Spectra,LipidIDSpectra ,BLSpectra, WaterSpectra, index, SingleMetabSpectra, FreqShift, PhShift, PeakWidth_Lor,PeakWidth_Gau, SNR, AcquDelay,LipidScaling,BLScaling,WaterScaling



