import matplotlib.pyplot as plt
import os, io, pickle
import h5py
import pandas as pd
import numpy as np
from numpy import array as a
import csv
import glob
import sys
import time
from functools import partial

#Robust script.

#_______________________________________________________________________________________________
#_______________________________________________________________________________________________
#_______________________________________________________________________________________________


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

#_______________________________________________________________________________________________
#_______________________________________________________________________________________________
#_______________________________________________________________________________________________


## added BL_spectra in loaded data
def load_data3(filename,
               load=('train/spectra', 'train/amplitudes', 'train/spectra/max', 'train/spectra/energy'),
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
        #print('k: {}'.format(k))
        #print('splits: {}'.format(k))
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

#_______________________________________________________________________________________________
#_______________________________________________________________________________________________
#_______________________________________________________________________________________________


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
    
    if 'realData' in arrays.keys():
        Data_rf=np.transpose(arrays['realData']+ 1j* arrays['imagData'],(1,0))
    else:
        Data_rf=arrays['Lipid_Stack_File']
    #Data_rf=arrays['realData']+ 1j* arrays['imagData']
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

#_______________________________________________________________

