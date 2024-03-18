#!/usr/bin/python3
import os
import tools
import Simulation
import h5py
import numpy as np
import matplotlib.pyplot as plt
#__________________________________________________________________________________________________________________________________________-
#Option for the running Launch
#__________________________________________________________________________________________________________________________________________-

def main():

#Generation of training and test dataset.
#Robuste Script (Peter's script majority re-use)

    import argparse
    parser = argparse.ArgumentParser(description='Generate dataset.')
    parser.add_argument('-o', '--output', dest='filename', required=True,nargs=1, help='Ouput filename (filename.h5)')
    parser.add_argument('-i', '--LipData', dest='InputFilenames', required=True,nargs='+', help='Input filenames (InFile1.h5 InFile2.h5 InFile3.h5)')
    parser.add_argument('-f', '--force', dest='update', action='store_true',
                        help='Force overwriting existing file')   

    args = parser.parse_args()
    filename = args.filename
    InputFiles = args.InputFilenames

	#__________________________________________________________________________________________________________________________________________-
	#What to do if the file already exists
	#__________________________________________________________________________________________________________________________________________-
    import os
    if os.path.isfile(filename[0]):
        if not args.update:
            print('{} already exists, add -f to force overwriting'.format(filename[0]))
            return
        else:
            os.remove(filename[0])
    print('Loading training dataset...')
    spectra_TotLip = []
    spectra_TotAll = []
    spectra_TotIDLip = []
    freq_shiftTot = []
    ph_shiftTot = []
    peakwidth_LorTot = []
    peakwidth_GauTot = []
    acqudelayTot = []
    LipidScalingTot = []
    BLScalingTot = []
    WaterScalingTot = []
    amplitudesTot = []

    
    for f in range(len(InputFiles)):
        print("InputFiles= ", InputFiles[f])
        #added SNR here but not in tools.py
        data_train = tools.load_data3(InputFiles[f],
                                load=('train/spectra','train/Lipid_BL_Wat_spectra','train/LipidID_spectra','train/freqshift','train/phshift','train/Lorwidth',
                                     'train/Gauwidth','train/acqdelay','train/LipidScaling','train/BLScaling','train/WatScaling','train/amplitudes'
                                     ))
    # meta-data of each array is loaded automatically with
        spectra_Lip = data_train['train/Lipid_BL_Wat_spectra'][:]
        spectra_All = data_train['train/spectra'][:]
        spectra_IDLip = data_train['train/LipidID_spectra'][:]
        freq_shift = data_train['train/freqshift'][:]
        phshift = data_train['train/phshift'][:]
        Lorwidth = data_train['train/Lorwidth'][:]
        Gauwidth = data_train['train/Gauwidth'][:]
        acqdelay = data_train['train/acqdelay'][:]
        LipidScaling = data_train['train/LipidScaling'][:]
        BLScaling = data_train['train/BLScaling'][:]        
        WatScaling = data_train['train/WatScaling'][:]
        amplitudes = data_train['train/amplitudes'][:]

        
        if f==0:
            spectra_TotLip=spectra_Lip
            spectra_TotAll=spectra_All
            spectra_TotIDLip=spectra_IDLip
            freq_shiftTot = freq_shift
            LorwidthTot = Lorwidth 
            GauwidthTot = Gauwidth 
            acqdelayTot = acqdelay 
            LipidScalingTot = LipidScaling 
            BLScalingTot = BLScaling 
            WatScalingTot = WatScaling 
            phshiftTot = phshift 
            amplitudesTot = amplitudes            
        else:
            spectra_TotLip = np.concatenate((spectra_TotLip,spectra_Lip),axis=0) 
            spectra_TotAll = np.concatenate((spectra_TotAll,spectra_All),axis=0)
            spectra_TotIDLip  = np.concatenate((spectra_TotIDLip,spectra_IDLip),axis=0)
            freq_shiftTot = np.concatenate((freq_shiftTot,freq_shift),axis=0)
            LorwidthTot = np.concatenate((LorwidthTot,Lorwidth),axis=0) 
            GauwidthTot = np.concatenate((GauwidthTot,Gauwidth),axis=0) 
            acqdelayTot = np.concatenate((acqdelayTot,acqdelay),axis=0) 
            LipidScalingTot = np.concatenate((LipidScalingTot,LipidScaling),axis=0) 
            BLScalingTot = np.concatenate((BLScalingTot,BLScaling),axis=0) 
            WatScalingTot = np.concatenate((WatScalingTot,WatScaling),axis=0) 
            phshiftTot = np.concatenate((phshiftTot,phshift),axis=0) 
            amplitudesTot = np.concatenate((amplitudesTot,amplitudes),axis=0)                    
              
    print("shape(spectra_TotLip)= {}".format(np.shape(spectra_TotLip)))
    
    data_train = tools.load_data3(InputFiles[0],
                                load=('train/spectra',
                                	'train/Lipid_BL_Wat_spectra','train/LipidID_spectra',
                                    'train/amplitudes',
                                    'train/spectra/max',
                                    'train/spectra/energy',
                                    'train/index','train/SNR',
                                    'train/spectra/MaxBLScaling','train/spectra/MaxWatScaling','train/spectra/MaxLipidScaling',
                                    'train/SingleMetab_spectra'
                                     ))    
    Fs=data_train['train/spectra/Fs'] 
    Npt=data_train['train/spectra/Npt']
    NMRFreq=data_train['train/spectra/NMRFreq']
    WINDOW_START=data_train['train/spectra/WINDOW_START']
    WINDOW_END=data_train['train/spectra/WINDOW_END']
    N1=data_train['train/spectra/N1']
    N2=data_train['train/spectra/N2']
    MaxLipidScaling=data_train['train/spectra/MaxBLScaling']
    MaxBLScaling=data_train['train/spectra/MaxWatScaling']
    MaxWatScaling=data_train['train/spectra/MaxLipidScaling']

    
    spectra_clean_max = data_train['train/spectra/max']
    spectra_clean_energy = data_train['train/spectra/energy'] 
    #BL_spectra = data_train['train/BL_spectra']
    index = data_train['train/index']
    names = data_train['train/names']
    names = names + ['SNR']
    snr = data_train['train/SNR']
    SingleMetab_spectra=data_train['train/SingleMetab_spectra']
    
    spectra_max = np.abs(spectra_TotAll[:]).max()
    spectra_energy = (np.abs((spectra_TotAll[:,:]) ) ** 2).sum(axis=1).mean()
   
    #LipOp_cff, Fs, Npt, N1, N2,NMRFreq =tools.load_LipidOpdata(LipProjFile)
    #LipOp_cff = np.array(LipOp_cff)
    #print('shape of LipOp_cff ={}.'.format(LipOp_cff.shape))     

    #Plot an exemple
    #plt.plot(range(N2-N1) , np.real(spectra[0,:])/Npt, '-')
    #plt.xlabel("Samples")
    #plt.ylabel("Spectral data")
    #plt.title("Exemple of Input data")
    #plt.show()
    print("Write to file ", filename[0])
    with h5py.File(filename[0], 'w-', libver='earliest') as f:
        train = f.create_group('train')
        dset = train.create_dataset('spectra', data=spectra_TotAll, fletcher32=True)
        dset.attrs['Fs'] = Fs
        dset.attrs['Npt'] = Npt
        dset.attrs['max'] = spectra_max
        dset.attrs['NMRFreq'] = NMRFreq
        dset.attrs['WINDOW_START'] = WINDOW_START
        dset.attrs['WINDOW_END'] = WINDOW_END
        dset.attrs['N1'] = N1
        dset.attrs['N2'] = N2
        dset.attrs['MaxLipidScaling'] = MaxLipidScaling
        dset.attrs['MaxBLScaling'] = MaxBLScaling
        dset.attrs['MaxWatScaling'] = MaxWatScaling
        dset.attrs['energy'] = spectra_energy
        train.create_dataset('SNR', data=snr) 
        #dset.attrs['SNR'] = snr
        train.create_dataset('freqshift', data=freq_shiftTot, fletcher32=True)  
        #dset.attrs['freqshift'] = freq_shift
        train.create_dataset('phshift', data=phshiftTot, fletcher32=True)  
        #dset.attrs['phshift'] = ph_shift
        train.create_dataset('Lorwidth', data=LorwidthTot, fletcher32=True)  
        #dset.attrs['Lorwidth'] = peakwidth_Lor
        train.create_dataset('Gauwidth', data=GauwidthTot, fletcher32=True)
        #dset.attrs['Gauwidth'] = peakwidth_Gau
        train.create_dataset('acqdelay', data=acqdelayTot, fletcher32=True)
        #dset.attrs['acqdelay'] = acqudelay
        train.create_dataset('LipidScaling', data=LipidScalingTot, fletcher32=True)
        train.create_dataset('BLScaling', data=BLScalingTot, fletcher32=True)
        train.create_dataset('WatScaling', data=WatScalingTot, fletcher32=True)  
        train.create_dataset('index', data=index, fletcher32=True)  
        train.create_dataset('SingleMetab_spectra', data=SingleMetab_spectra)     
        train.create_dataset('amplitudes', data=amplitudesTot, fletcher32=True)

        train.create_dataset('Lipid_BL_Wat_spectra', data=spectra_TotLip, fletcher32=True)
        train.create_dataset('LipidID_spectra', data=spectra_TotIDLip, fletcher32=True)
       


if __name__ == '__main__':
    main()
