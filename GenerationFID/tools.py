import os
import h5py
import numpy as np
from numpy import array as a
import sys
import time

#Robust script.

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
    with h5py.File(filename, 'r', libver='earliest') as f:
        for k, v in f.items():
             arrays[k] = np.array(v)

    samplerate=arrays['samplerate']
    Npt=int(arrays['Npt'])
    N1=int(arrays['N1'])
    N2=int(arrays['N2'])
    NMRfreq=arrays['NMRfreq']
    
    if 'realData' in arrays.keys():
        Data_rf=np.transpose(arrays['realData']+ 1j* arrays['imagData'],(1,0))
    else:
        Data_rf=arrays['Lipid_Stack_File']
    return Data_rf,  samplerate, Npt, N1, N2,NMRfreq
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
    return Data_cff,  samplerate, Npt, N1, N2,NMRfreq

#_______________________________________________________________

