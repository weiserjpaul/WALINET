#!/usr/bin/python3


def main():
    import numpy as np

    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import tools
    import sys
    from functools import partial
    if len(sys.argv) < 2:
        print('usage: ./show_data_sample.py data_filename.h5')
        sys.exit(1)
   
    arrays = {}
    import h5py
    import numpy as np
    
    
 
    data = tools.load_data3(sys.argv[1], load=('train/spectra',
                                	'train/Lipid_BL_Wat_spectra','train/LipidID_spectra','train/Metab_spectra',
                                    'train/amplitudes','train/LipidScaling', 'train/BLScaling', 
                                    'train/spectra/max',
                                    'train/spectra/energy',
                                    'train/index','train/SNR','train/Lorwidth','train/Gauwidth','train/freqshift','train/phshift','train/acqdelay' 
                                     ))
    
    Fs=data['train/spectra/Fs'] 
    Npt=data['train/spectra/Npt']
    NMRFreq=data['train/spectra/NMRFreq']
    WINDOW_START=data['train/spectra/WINDOW_START']
    WINDOW_END=data['train/spectra/WINDOW_END']
    N1=data['train/spectra/N1']
    N2=data['train/spectra/N2']

    spectra_clean_max = data['train/spectra/max']
    spectra_clean_energy = data['train/spectra/energy'] 
    index = data['train/index']
    
    #NMRFreq = 123.2625 * 1e6 
    #N1 = int(np.around(((4.7-tools.WINDOW_START) * 1e-6 * NMRFreq)* Npt / Fs))
    #N2 = int(np.around(((4.7-tools.WINDOW_END) * 1e-6 * NMRFreq)* Npt / Fs))
    #print('Training between point {} and point {} from total length {} at {} Hz BW.'.format(N1,N2,Npt,Fs))
    spectra_All = data['train/spectra'][:]
    spectra_IDLip = data['train/LipidID_spectra'][:]
    spectra_LipBLWat = data['train/Lipid_BL_Wat_spectra'][:]
    spectra_Metab = data['train/Metab_spectra'][:]

    spectra_IDMetab = spectra_All - spectra_IDLip
    # amplitudes = data['amplitudes']
    # names = data['names']
    snr = data['train/SNR']
    LipidScaling = data['train/LipidScaling']
    BLScaling = data['train/BLScaling']
    global fig,ax, k0
    fig, ax = plt.subplots(5)
    k0 = 0
    def draw_spectra():
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()
        ax[0].plot(np.real(spectra_All[k0, :].T))
        ax[1].plot(np.imag(spectra_All[k0, :].T),label='All Included') 
        ax[0].plot(np.real(spectra_IDLip[k0, :].T))
        ax[1].plot(np.imag(spectra_IDLip[k0, :].T),label='Op IDied Lipids') 
        ax[0].plot(np.real(spectra_LipBLWat[k0, :].T))
        ax[1].plot(np.imag(spectra_LipBLWat[k0, :].T),label='Simulated Lip, Water and BL') 

        ax[0].set_title('Real')
        ax[1].set_title('Imag')
        ax[1].legend()
        ax[2].text(0,0,'LipidScaling = {:1.1f},BLScaling = {:1.1f}, SNR = {:1.1f}, Width = {:1.1f}, \n FreqShift = {:1.1f}, PhaseShift = {:1.1f}, AcqDel = {:1.4f} '.format( data['train/LipidScaling'][k0,0],data['train/BLScaling'][k0,0] ,data['train/SNR'][k0,0],(data['train/Lorwidth'][k0,0]+data['train/Gauwidth'][k0,0]),data['train/freqshift'][k0,0],data['train/phshift'][k0,0],data['train/acqdelay'][k0,0]))
        ax[2].get_xaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)
 
 
        ax[3].clear()
        ax[4].clear()
        ax[3].plot(np.real(spectra_Metab[k0, :].T))
        ax[4].plot(np.imag(spectra_Metab[k0, :].T),label='Simulated Metabolite') 
        ax[3].plot(np.real(spectra_IDMetab[k0, :].T))
        ax[4].plot(np.imag(spectra_IDMetab[k0, :].T),label='Op IDied Metabolites') 
        ax[3].set_title('Real')
        ax[4].set_title('Imag')
        ax[4].legend()

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
