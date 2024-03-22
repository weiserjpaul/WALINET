

import os, io
import h5py

import pandas as pd
import numpy as np
from numpy import array as a
import csv
import glob
import sys
import time
import re


def ReadModeFiles(index,list_file):
    NbM = len(list_file)
    temp_modes = [None] * NbM  # empty list of size NbM 
    for i, filename in enumerate(list_file):
        metabo_mode = pd.read_csv(filename, header=None, skiprows=[0]).values
        m = re.search("[0-9]T_.{1,6}_Exact", filename)
        name = bytes(filename[m.span()[0]+3:m.span()[1]-6].strip(), 'utf8')
        temp_modes[index[name]] = metabo_mode
    return temp_modes
    
def SimulateTrainingData(NbEx, MaxSNR, MinSNR, MaxFreq_Shift,MaxPeak_Width, MinPeak_Width, NbBL,MinBLWidth, MaxAcquDelay, MaxLipidScaling,MaxBLScaling,MaxWatScaling,NbWat,LipStackFile='',LipOpFile='', verbose=True ):
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
    if LipStackFile:  
        Lipid_rf, Fs, Npt, N1_lip, N2_lip, NMRFreq =tools.load_LipidStackdata(LipStackFile)
        print('shape of Lipid_rf ={}.'.format(Lipid_rf.shape))
        print("Nb of Nan in Lipid_rf = ", np.sum(np.isnan(Lipid_rf[:])))
    
    LipOp_cff, Fs, Npt, N1, N2,NMRFreq =tools.load_LipidOpdata(LipOpFile)
    LipOp_cff = np.array(LipOp_cff)
    print('shape of LipOp_cff ={}.'.format(LipOp_cff.shape))
    print("Nb of Nan in LipOp_cff = ", np.sum(np.isnan(LipOp_cff[:])))

    if verbose:
        print('Generating data ... ')
    

    
    N = Npt

    TimeSerie = np.zeros(( N), dtype=np.complex64)
    TimeSerieClean = np.zeros(( N), dtype=np.complex64)
    LipTimeSerie = np.zeros((N), dtype=np.complex64)
    BLTimeSerie = np.zeros(( N), dtype=np.complex64)
    WaterTimeSerie = np.zeros(( N), dtype=np.complex64)
    TempSpectrum = np.zeros(( N), dtype=np.complex64)

    AllInSpectra = np.zeros((NbEx, (N2-N1)), dtype=np.complex64)
    MetabSpectra = np.zeros((NbEx, (N2-N1)), dtype=np.complex64)
    LipidIDSpectra = np.zeros((NbEx, (N2-N1)), dtype=np.complex64)
    Lipid_BL_Wat_Spectra = np.zeros((NbEx, (N2-N1)), dtype=np.complex64)
    BLSpectra = np.zeros((NbEx, (N2-N1)), dtype=np.complex64)
    WaterSpectra = np.zeros((NbEx, (N2-N1)), dtype=np.complex64)
    Lipid_rf_Rand = np.zeros((NbEx, (N)), dtype=np.complex64)
    Time = np.linspace(0, N - 1, N) / Fs 
    Frequencies = np.linspace(0, 1, N) * Fs


    index = {}  # mapping of metabolite name to index
    ScriptPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))    
    mean_std_csv = pd.read_csv(ScriptPath+'/MetabModes/Metab_Mean_STD.txt', header=None).values

    for i, v in enumerate(mean_std_csv[:, 0].astype(str)):
        index[ bytes(v.strip(), 'utf8') ] = i
        
    mean_std = mean_std_csv[:, 1:].astype(np.float32)
  
    list_file = glob.glob(ScriptPath+'/MetabModes/3T_TE0/*Exact_Modes.txt')
    NbM = len(list_file)

    metabo_modes = [[[None] for j in range(NbM)] for i in range(6)]
    temp_modes = ReadModeFiles(index,list_file)
    metabo_modes[0]=temp_modes
    
    list_file = glob.glob(ScriptPath+'/MetabModes/3T_TE47/*Exact_Modes.txt')
    if (NbM != len(list_file)):
        print('ERROR: {} metabolite found in 3T_TE47 but {} in the system. Stopping the simuation!'.format(len(list_file),NbM))      
        return   
    temp_modes = ReadModeFiles(index,list_file)
    metabo_modes[1]=temp_modes
    
    list_file = glob.glob(ScriptPath+'/MetabModes/3T_TE95/*Exact_Modes.txt')
    if (NbM != len(list_file)):
        print('ERROR: {} metabolite found in 3T_TE95 but {} in the system. Stopping the simuation!'.format(len(list_file),NbM))      
        return   
    temp_modes = ReadModeFiles(index,list_file)
    metabo_modes[2]=temp_modes   
    
    list_file = glob.glob(ScriptPath+'/MetabModes/7T_TE0/*Exact_Modes.txt')
    if (NbM != len(list_file)):
        print('ERROR: {} metabolite found in 7T_TE0 but {} in the system. Stopping the simuation!'.format(len(list_file),NbM))      
        return   
    temp_modes = ReadModeFiles(index,list_file)
    metabo_modes[3]=temp_modes   
    
    list_file = glob.glob(ScriptPath+'/MetabModes/7T_TE38/*Exact_Modes.txt')
    if (NbM != len(list_file)):
        print('ERROR: {} metabolite found in 7T_TE38 but {} in the system. Stopping the simuation!'.format(len(list_file),NbM))      
        return   
    temp_modes = ReadModeFiles(index,list_file)
    metabo_modes[4]=temp_modes    
    
    list_file = glob.glob(ScriptPath+'/MetabModes/7T_TE76/*Exact_Modes.txt')
    if (NbM != len(list_file)):
        print('ERROR: {} metabolite found in 7T_TE76 but {} in the system. Stopping the simuation!'.format(len(list_file),NbM))      
        return   
    temp_modes = ReadModeFiles(index,list_file)
    metabo_modes[5]=temp_modes
    
    
    
    if verbose:
        print(index)
        print(mean_std)
    from numpy.random import rand, randn, seed
    seed()
 

    Amplitude = mean_std[:, 1]* randn(NbEx, NbM) + mean_std[:, 0]
    # putting negative values to 0
    Amplitude = Amplitude.clip(min=0)

    FreqShift = (rand(NbEx, 1)*2 - 1) * MaxFreq_Shift
    PeakWidth = MinPeak_Width + rand(NbEx, 1) * (MaxPeak_Width - MinPeak_Width)
    ponder_peaks = rand(NbEx, 1)
    PeakWidth_Gau = np.multiply(ponder_peaks, PeakWidth)
    PeakWidth_Lor = np.multiply(1-ponder_peaks, PeakWidth)
    
    PeakWidth = rand(NbEx, 1) * 10;
    LipWidth_Gau = np.multiply(ponder_peaks, PeakWidth)
    LipWidth_Lor = np.multiply(1-ponder_peaks, PeakWidth)
    LipFreqShift = (rand(NbEx, 1)*2 - 1) * 10;

    BasisI = np.floor(rand(NbEx, 1) * 6)
    PhShift = rand(NbEx, 1) * 2 * PI
    AcquDelay =  (rand(NbEx, 1)-0.5)*2 * MaxAcquDelay
    LipidScaling =  1e-1*(10**(rand(NbEx, 1)*np.log10(1e1*MaxLipidScaling)))   
    
    WaterScaling=  1e-2*(10**(rand(NbEx, 1)*np.log10(1e2*MaxWatScaling)))
    BLScaling =  1e-2*(10**(rand(NbEx, 1)*np.log10(1e2*MaxBLScaling)))
    
    
    LipidPh = rand(NbEx, 1) * 2 * PI

    NbCombVox = 10
    if LipStackFile:    
        for ex in range(NbEx):
            LipPos=np.round((rand(NbCombVox)*np.shape(Lipid_rf)[0]-0.5)*0.9999999999)
            
            LipAmp=(rand(NbCombVox));
            LipidPh= 2 * PI * rand(NbCombVox)
            LipAmp=LipAmp/np.sum(LipAmp)
            Lipid_rf_Rand[ex,:] = 0
            for LipV in range(NbCombVox):
                Lipid_rf_Rand[ex,:] += LipAmp[LipV]*np.exp(1j * LipidPh[LipV])*Lipid_rf[int(LipPos[LipV]),:]
            
    

    SNR = MinSNR + rand(NbEx, 1) * (MaxSNR - MinSNR)
    


    
    print('len(metabo_modes): {}'.format(len(metabo_modes)))
    print('np.shape(Amplitude): {}'.format(np.shape(Amplitude))) 
    
    BatchSize = int(NbEx/200)+1
    print('Data generation Batch size: ({} ).'.format(BatchSize))
    LipidIDSpBatch = np.zeros((BatchSize, (N)), dtype=np.complex64)

    LipCoilInd = 0 
    BatchI = 0
    BlockI = 0
    TempMetabData = np.zeros( (len(metabo_modes[0]), N), dtype=np.complex64)
    
    for ex in range(NbEx):
    
        TempMetabData =0*TempMetabData
        for f, mode in enumerate(metabo_modes[int(BasisI[ex])]):
                Freq = ((4.7-mode[:, 0]) * 1e-6 * NMRFreq)[...,None]
                for Nuc in range(len(Freq)):
                    if (mode[Nuc, 0] > 0.0) & (mode[Nuc, 0] < 4.5)  : # only for the window of interest 
                        TempMetabData[f, :] += mode[Nuc, 1][...,None] * np.exp(1j * mode[Nuc, 2][...,None]) * np.exp(2 * np.pi * 1j * (Time + AcquDelay[ex])  * Freq[Nuc])
        
        
        if verbose:
            if np.mod(ex, int(NbEx / 100)) == 0:
                print('{}% '.format(int(ex * 100 / NbEx)), end='', flush=True)    
        
        TimeSerieClean=0*TimeSerieClean
        for f, _ in enumerate(metabo_modes[int(BasisI[ex])]):
            TimeSerieClean[:] += Amplitude[ex, f] * TempMetabData[f, :]* np.exp(1j * PhShift[ex])  
            
        TimeSerieClean[:] = TimeSerieClean[ :] * np.exp( (Time* 1j * 2 * PI * FreqShift[ex] ) + (- (np.square(Time)) * (np.square(PeakWidth_Gau[ex]))) + ((np.absolute(Time)) * (- PeakWidth_Lor[ex]) ) )
        SpectrumTemp = np.fft.fft(TimeSerieClean[:],axis=0) 

        NCRand=(randn(N) + 1j * randn(N))
        TimeSerie[:] = TimeSerieClean[:] + np.fft.ifft(SpectrumTemp[N1:N2].std()/0.65 / SNR[ex] * NCRand,axis=0)


        BLTimeSerie=0*BLTimeSerie
        for ga in range(NbBL):
            AmpGauss=rand(1)
            TimeGauss=rand(1)/MinBLWidth
            TimeLor=rand(1)/MinBLWidth
            PPMG=6*rand(1)-1 # between 5 and -1 
            FreqGauss=- ( PPMG - 4.7 ) * 1e-6 * NMRFreq
            PhGaus= 2 * PI * rand(1)
            BLTimeSerie[:] += AmpGauss*np.exp(1j*PhGaus + (Time) * 2 * PI * 1j* FreqGauss -Time**2 /(TimeGauss**2) -Time /(TimeLor))
            
        WaterTimeSerie=0*WaterTimeSerie
        for wat in range(NbWat):
            GaussWidth=((rand(1)*0.4)+0.8)*PeakWidth_Gau[ex] # PeakWidth_Gau +/- 20%
            LorWidth=((rand(1)*0.4)+0.8)*PeakWidth_Lor[ex] # PeakWidth_Lor +/- 20%
            PhWater= 2 * PI * rand(1)
            FreqWater = (rand(1)*2-1)*10+ FreqShift[ex] # FreqShift +/- 10Hz
            MecFreq= NMRFreq*1e-6*(0.5+rand(1)*2.5) #4.2-1.2ppm
            MecWidth = 10+rand(1)*40 #10-50Hz
            FreqSB=rand(1)*np.sin(2*PI*MecFreq*Time+rand(1)*2*PI)*np.exp(-Time *MecWidth); #SideBand Freq (function of time!)
            WaterTimeSerie[:] += rand(1)*np.exp(1j*PhWater + (Time) * 2 * PI * 1j* (FreqWater + FreqSB) -Time**2 *(GaussWidth**2) -Time *(LorWidth))



        Metab_max = np.max(np.abs(np.fft.fft(TimeSerie, axis=0)[N1:N2]), axis=0)
  
        if NbBL>0:
            BL_max =  np.max(np.abs(np.fft.fft(BLTimeSerie, axis=0)[N1:N2]), axis=0)
            BLSpectra[ex,:] = np.fft.fft(BLTimeSerie, axis=0)[N1:N2]*Metab_max/BL_max*BLScaling[ex]
        else:
            BLSpectra[ex,:] = 0

        Water_max =  np.max(np.abs(np.fft.fft(WaterTimeSerie, axis=0)[N1:N2]), axis=0)
        WaterSpectra[ex,:] = np.fft.fft(WaterTimeSerie, axis=0)[N1:N2]*Metab_max/Water_max*WaterScaling[ex]
        if LipStackFile:   
            Lip_max =  np.max(np.abs(Lipid_rf_Rand[ex,N1:N2]), axis=0)
        else:
            Lip_max = 1E32

        LipTimeSerie[:] = np.fft.ifft(Lipid_rf_Rand[ex,:] , axis=0)

        LipTimeSerie *= np.exp(-(Time *LipWidth_Gau[ex])**2 -Time*LipWidth_Lor[ex])
        LipTimeSerie *= np.exp( Time* 1j * 2 * PI * LipFreqShift[ex] )
        
        #Spectra with Full Lip, Water and metab
        Lipid_BL_Wat_Spectra[ex, :] =  BLSpectra[ex,:] + WaterSpectra[ex,:] +  Metab_max/Lip_max*LipidScaling[ex] *np.fft.fft(LipTimeSerie[:], axis=0)[N1:N2] 

        #Spectra with Removed Lip, Water and metab
        
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

        

    SingleMetabSpectra = np.fft.fft(TempMetabData, axis=1)[:,N1:N2]

    if verbose:
        print('100%', flush=True)
    return AllInSpectra, Amplitude, MetabSpectra,Lipid_BL_Wat_Spectra,LipidIDSpectra ,BLSpectra, WaterSpectra, index, SingleMetabSpectra, FreqShift, PhShift, PeakWidth_Lor,PeakWidth_Gau, SNR, AcquDelay,LipidScaling,BLScaling,WaterScaling

