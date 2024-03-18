#!/usr/bin/python3

#__________________________________________________________________________________________________________________________________________-
#Option for the running Launch
#__________________________________________________________________________________________________________________________________________-

def main(raw_args=None):

#Generation of training and test dataset.
#Robuste Script (Peter's script majority re-use)

    import argparse
    parser = argparse.ArgumentParser(description='Generate dataset.')
    parser.add_argument('-o', '--output', dest='filename', required=True, help='Ouput filename (filename.h5)')
    parser.add_argument('-l', '--LipData', dest='LipidFile', help='Lipid Data filename (LipidFile.h5)')
    parser.add_argument('-lop', '--LipOp', dest='LipidProjFile',required=True)
    #parser.add_argument('-w', '--WaterData', dest='WaterFile',default=0, help='Water Data filename (WFile.h5)')
    #parser.add_argument('--wstart', dest='WINDOW_START', default=4.5,
    #                    help='highest ppm value of the frequency window (default: %(default)s)')
    #parser.add_argument('--wend', dest='WINDOW_END', default=0.0,
    #                    help='lowest ppm value of the frequency window (default: %(default)s)')
    #parser.add_argument('-N1', dest='N1', required=True, help='First point of the Freq. window (default: %(default)s)')
    #parser.add_argument('-N2', dest='N2', required=True, help='Last point of the Freq. window  (default: %(default)s)')
    
    parser.add_argument('--ntrain', dest='nbex_train', default=1000000,
                        help='Number of realizations for training (default: %(default)s)')
    parser.add_argument('--ntest', dest='nbex_test', default=1000,
                        help='Number of realizations for testing (default: %(default)s)')
    parser.add_argument('--minsnr', dest='MinSNR', default=1,
                        help='Minimal Signal to Noise Ratio (default: %(default)s)')
    parser.add_argument('--maxsnr', dest='MaxSNR', default=8,
                        help='Maximal Signal to Noise Ratio (default: %(default)s)')
                        
    parser.add_argument('--maxFShft', dest='MaxFreq_Shift', default=40.0,
                        help='Maximal Frequency Shift (default: %(default)s)')
    parser.add_argument('--maxPkW', dest='MaxPeak_Width', default=20.0,
                        help='Maximum peak width (FWHM*pi) (default: %(default)s)')
    parser.add_argument('--minPkW', dest='MinPeak_Width', default=4.0,
                        help='Minimum peak width (FWHM*pi) (default: %(default)s)')

    parser.add_argument('--AcqDel', dest='MaxAcqDelay', default=0.002,
                        help='Maximum realized acquisition delay in s (default: %(default) s)')
    parser.add_argument('--maxLipSc', dest='MaxLipidScaling', default=400.0,
			help='Maximum relative lipid to metabolite amplitudes (default: %(default))')
    parser.add_argument('--maxBLSc', dest='MaxBLScaling', default=2.0,
			help='Maximum relative BL  to metabolite amplitudes (default: %(default))')
			
    parser.add_argument('--nBL', dest='NbBL', default=10,
                        help='Number of Gaussian functions in the Baseline (default: %(default))')
    parser.add_argument('--wBL', dest='MinBLWidth', default=200,
                        help='Min peak width of Baseline composents (default: %(default))')
                        
    parser.add_argument('--maxWatSc', dest='MaxWatScaling', default=100.0,
			help='Maximum relative BL  to metabolite amplitudes (default: %(default))')
    parser.add_argument('--nWat', dest='NbWat', default=10,
                        help='Number of peak functions to simulate the water signal(default: %(default))')                     			
    #parser.add_argument('--nmrfreq', dest='NMRFreq', default='123.2625e6',
                        #help='Magnet Resonance Freq. (default: %(default) s)')
    parser.add_argument('-f', '--force', dest='update', action='store_true',
                        help='Force overwriting existing file')
    parser.add_argument('-v', dest='verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args(raw_args)
    filename = args.filename

	#__________________________________________________________________________________________________________________________________________-
	#What to do if the file already exists
	#__________________________________________________________________________________________________________________________________________-
    print(args)
    import os
    if os.path.isfile(filename):
        if not args.update:
            print('{} already exists, add -f to force overwriting'.format(filename))
            return
        else:
            os.remove(filename)
    import tools
    import Simulation
    import h5py
    import numpy as np
    #import Constante
    import matplotlib.pyplot as plt

    #Lipid_rf, samplerate, Npt, N1, N2,NMRFreq =tools.load_LipidStackdata(args.LipidFile)
    LipOp_cff, Fs, Npt, N1, N2,NMRFreq =tools.load_LipidOpdata(args.LipidProjFile)

    print('np.shape(LipOp_cff): {}'.format(np.shape(LipOp_cff)))
    print('Npt: {}'.format(Npt))
    print('Fs: {}'.format(Fs))
    print('N1: {}'.format(N1))
    print('N2: {}'.format(N2))
    print('NMRFreq: {}'.format(NMRFreq))
    #Npt = int(args.Npt)
    #WINDOW_START = float(args.WINDOW_START)
    #WINDOW_END = float(args.WINDOW_END)
   # NMRFreq = float(args.NMRFreq)
    #N1 = int(np.around(((4.7-WINDOW_START) * 1e-6 * NMRFreq)* Npt / Fs))
    #N2 = int(np.around(((4.7-WINDOW_END) * 1e-6 * NMRFreq)* Npt / Fs))
    #N1 = int(args.N1)
    #N2 = int(args.N2)
    if np.mod(N2-N1,4) != 0 :
        print('N2-N1 is not a factor of 4. Incompatible data ' ) 
        return 0

    #N2=N2 + np.mod(N2-N1,4);
    #print('Rounding up N2 to (must be a factor of 4): N2= {}'.format(N2) ) 
    WINDOW_END = 4.7-N2*Fs/Npt/NMRFreq*1e6
    WINDOW_START = 4.7-N1*Fs/Npt/NMRFreq*1e6
    #print('Rounding up WINDOW_END then to {}'.format(WINDOW_END))
    #__________________________________________________________________________________________________________________________________________-
    #Generate the training set
    #__________________________________________________________________________________________________________________________________________-

    if args.verbose:
        print('Generating training data ({} realizations)'.format(args.nbex_train))



    AllInSpectra, amplitudes, Metab_spectra, Lipid_BL_Wat_spectra,LipidID_spectra,BL_spectra,Water_spectra, index, SingleMetab_spectra, freq_shift, ph_shift,  peakwidth_Lor,peakwidth_Gau, snr, acqudelay,LipidScaling,BLScaling,WaterScaling = Simulation.SimulateTrainingData(NbEx=int(args.nbex_train), MaxSNR=float(args.MaxSNR), MinSNR=float(args.MinSNR), MaxFreq_Shift=float(args.MaxFreq_Shift), MaxPeak_Width=float(args.MaxPeak_Width),MinPeak_Width=float(args.MinPeak_Width), NbBL=int(args.NbBL), MinBLWidth=float(args.MinBLWidth), MaxAcquDelay=float(args.MaxAcqDelay), MaxLipidScaling=float(args.MaxLipidScaling),MaxBLScaling=float(args.MaxBLScaling),MaxWatScaling=float(args.MaxWatScaling),NbWat=int(args.NbWat), LipStackFile=args.LipidFile, LipOpFile=args.LipidProjFile , verbose=args.verbose)


    print('Data Energy calculated between point {} and point {} from total length {}.'.format(N1,N2,Npt))
    index = list(index.items())

    spectra_max = np.abs(AllInSpectra[:]).max()
    spectra_energy = (np.abs((AllInSpectra[:,:]) ) ** 2).sum(axis=1).mean()
   
    #LipOp_cff, Fs, Npt, N1, N2,NMRFreq =tools.load_LipidOpdata(LipProjFile)
    #LipOp_cff = np.array(LipOp_cff)
    #print('shape of LipOp_cff ={}.'.format(LipOp_cff.shape))     

    #Plot an exemple
    #plt.plot(range(N2-N1) , np.real(spectra[0,:])/Npt, '-')
    #plt.xlabel("Samples")
    #plt.ylabel("Spectral data")
    #plt.title("Exemple of Input data")
    #plt.show()

    with h5py.File(filename, 'w-', libver='earliest') as f:
        train = f.create_group('train')
        dset = train.create_dataset('spectra', data=AllInSpectra, fletcher32=True)
        dset.attrs['Fs'] = Fs
        dset.attrs['Npt'] = Npt
        dset.attrs['max'] = spectra_max
        dset.attrs['NMRFreq'] = NMRFreq
        dset.attrs['WINDOW_START'] = WINDOW_START
        dset.attrs['WINDOW_END'] = WINDOW_END
        dset.attrs['N1'] = N1
        dset.attrs['N2'] = N2
        dset.attrs['MaxLipidScaling'] = float(args.MaxLipidScaling)
        dset.attrs['MaxBLScaling'] = float(args.MaxBLScaling)
        dset.attrs['MaxWatScaling'] = float(args.MaxWatScaling)
        dset.attrs['energy'] = spectra_energy
        train.create_dataset('SNR', data=snr) 
        #dset.attrs['SNR'] = snr
        train.create_dataset('freqshift', data=freq_shift, fletcher32=True)  
        #dset.attrs['freqshift'] = freq_shift
        train.create_dataset('phshift', data=ph_shift, fletcher32=True)  
        #dset.attrs['phshift'] = ph_shift
        train.create_dataset('Lorwidth', data=peakwidth_Lor, fletcher32=True)  
        #dset.attrs['Lorwidth'] = peakwidth_Lor
        train.create_dataset('Gauwidth', data=peakwidth_Gau, fletcher32=True)
        #dset.attrs['Gauwidth'] = peakwidth_Gau
        train.create_dataset('acqdelay', data=acqudelay, fletcher32=True)
        #dset.attrs['acqdelay'] = acqudelay
        train.create_dataset('LipidScaling', data=LipidScaling, fletcher32=True)
        train.create_dataset('BLScaling', data=BLScaling, fletcher32=True)
        train.create_dataset('WatScaling', data=WaterScaling, fletcher32=True)  
        train.create_dataset('index', data=index, fletcher32=True)  
        train.create_dataset('SingleMetab_spectra', data=SingleMetab_spectra)     
        train.create_dataset('amplitudes', data=amplitudes, fletcher32=True)

        train.create_dataset('Metab_spectra', data=Metab_spectra, fletcher32=True)
        train.create_dataset('Lipid_BL_Wat_spectra', data=Lipid_BL_Wat_spectra, fletcher32=True)
        train.create_dataset('LipidID_spectra', data=LipidID_spectra, fletcher32=True)

        train.create_dataset('BL_spectra', data=BL_spectra, fletcher32=True)
        train.create_dataset('Water_spectra', data=Water_spectra, fletcher32=True)

    #__________________________________________________________________________________________________________________________________________-
    #Generate the test set
    #__________________________________________________________________________________________________________________________________________-

    if args.verbose:
        print('Generating testing data ({} realizations)'.format(args.nbex_test))

    AllInSpectra, amplitudes, Metab_spectra, Lipid_BL_Wat_spectra,LipidID_spectra,BL_spectra,Water_spectra, index, SingleMetab_spectra, freq_shift, ph_shift,  peakwidth_Lor,peakwidth_Gau, snr, acqudelay,LipidScaling,BLScaling,WaterScaling = Simulation.SimulateTrainingData( NbEx=int(args.nbex_test), MaxSNR=float(args.MaxSNR), MinSNR=float(args.MinSNR), MaxFreq_Shift=float(args.MaxFreq_Shift), MaxPeak_Width=float(args.MaxPeak_Width),MinPeak_Width=float(args.MinPeak_Width), NbBL=int(args.NbBL), MinBLWidth=float(args.MinBLWidth), MaxAcquDelay=float(args.MaxAcqDelay), MaxLipidScaling=float(args.MaxLipidScaling),MaxBLScaling=float(args.MaxBLScaling),MaxWatScaling=float(args.MaxWatScaling),NbWat=int(args.NbWat), LipStackFile=args.LipidFile, LipOpFile=args.LipidProjFile, verbose=args.verbose)
    
    index = list(index.items())
   
    # Create h5 file with write-fail-if-exists
    # libver=latest: latest fileformat, faster (less compatible)
    with h5py.File(filename, 'a', libver='earliest') as f:
        train = f.create_group('test')
        dset = train.create_dataset('spectra', data=AllInSpectra, fletcher32=True)
        dset.attrs['Fs'] = Fs
        dset.attrs['Npt'] = Npt
        dset.attrs['max'] = spectra_max
        dset.attrs['NMRFreq'] = NMRFreq
        dset.attrs['WINDOW_START'] = WINDOW_START
        dset.attrs['WINDOW_END'] = WINDOW_END
        dset.attrs['N1'] = N1
        dset.attrs['N2'] = N2
        dset.attrs['MaxLipidScaling'] = float(args.MaxLipidScaling)
        dset.attrs['MaxBLScaling'] = float(args.MaxBLScaling)
        dset.attrs['MaxWatScaling'] = float(args.MaxWatScaling)
        dset.attrs['energy'] = spectra_energy
        train.create_dataset('SNR', data=snr, fletcher32=True)
   
        #dset.attrs['SNR'] = snr
        train.create_dataset('freqshift', data=freq_shift, fletcher32=True)  
        #dset.attrs['freqshift'] = freq_shift
        train.create_dataset('phshift', data=ph_shift, fletcher32=True)  
        #dset.attrs['phshift'] = ph_shift
        train.create_dataset('Lorwidth', data=peakwidth_Lor, fletcher32=True)  
        #dset.attrs['Lorwidth'] = peakwidth_Lor
        train.create_dataset('Gauwidth', data=peakwidth_Gau, fletcher32=True)
        #dset.attrs['Gauwidth'] = peakwidth_Gau
        train.create_dataset('acqdelay', data=acqudelay, fletcher32=True)
        #dset.attrs['acqdelay'] = acqudelay
        train.create_dataset('LipidScaling', data=LipidScaling, fletcher32=True)
        train.create_dataset('BLScaling', data=BLScaling, fletcher32=True)
        train.create_dataset('WatScaling', data=WaterScaling, fletcher32=True) 
        train.create_dataset('index', data=index, fletcher32=True)  
        train.create_dataset('SingleMetab_spectra', data=SingleMetab_spectra, fletcher32=True)     
        train.create_dataset('amplitudes', data=amplitudes, fletcher32=True)

        #train.create_dataset('Metab_spectra', data=Metab_spectra, fletcher32=True)
        train.create_dataset('Lipid_BL_Wat_spectra', data=Lipid_BL_Wat_spectra, fletcher32=True)
        train.create_dataset('LipidID_spectra', data=LipidID_spectra, fletcher32=True)

        #train.create_dataset('BL_spectra', data=BL_spectra, fletcher32=True)
        #train.create_dataset('Water_spectra', data=Water_spectra, fletcher32=True)

if __name__ == '__main__':
    main()
