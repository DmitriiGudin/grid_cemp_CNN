# Created by Dmitrii Gudin (U Notre Dame).

from __future__ import division
import numpy as np
import h5py
import os
import time
import params
import random


log_freq = 1000 # How often (in terms of the number of files processed) to print logging information. Non-positive value - no logging.


def get_column(N, Type, filename): # Retrieves the data from the Nth column of a file.
    return np.genfromtxt(filename, dtype=Type, usecols=N, comments=None)


def convert_to_hdf5 (file_list):
    print "Files to convert:", str(len(file_list))
    # Create the h5py output file, removing the old one if exists. 
    if os.path.isfile(params.hdf5_file):
        os.remove(params.hdf5_file)
    hdf5_file = h5py.File(params.hdf5_file, 'w')
    # Create categories for all data elements.
    hdf5_file.create_dataset("/spectrum", (len(file_list), len(params.wavelength_list)), dtype='f')
    hdf5_file.create_dataset("/T_EFF", (len(file_list), 1), dtype='f')
    hdf5_file.create_dataset("/LOG_G", (len(file_list), 1), dtype='f')
    hdf5_file.create_dataset("/FE_H", (len(file_list), 1), dtype='f')
    hdf5_file.create_dataset("/C_FE", (len(file_list), 1), dtype='f')
    # Start time count for logging.
    time_begin = time.time()
    # Shuffle the indeces, if needed.
    if params.shuffle_flag==1:
        indeces = range(0,len(file_list))
        random.shuffle(indeces)
    # Loop over all files and convert the data.
    for f, n in zip (file_list, range(0, len(file_list))):
        hdf5_file["/spectrum"][indeces[n]] = get_column(1, float, f)
        hdf5_file["/T_EFF"][indeces[n]] = float(f[-25:-21])
        hdf5_file["/LOG_G"][indeces[n]] = float(f[-20:-16])
        hdf5_file["/FE_H"][indeces[n]] = float(f[-15:-10])
        hdf5_file["/C_FE"][indeces[n]] = float(f[-9:-4])
        if log_freq>0:
            if n % log_freq == 0:
                print str(int(time.time()-time_begin)), "s: Building the hdf5 file.", str(n), "files out of", str(len(file_list)), "processed."
    # Close the file.
    hdf5_file.close
    print str(int(time.time()-time_begin)), "s: Done."
        

if __name__ == '__main__':
    file_dir = params.file_dir # Folder containing the data.
    # Get the list of the subdirectories in the folder. Leave only those that start with T.
    subfolders = [s for s in os.listdir(file_dir) if s[0]=='T']
    # Retrieve the full file list.
    file_list = [file_dir+"/"+str(s)+"/"+f for s in subfolders for f in os.listdir(file_dir+"/"+str(s))]
    convert_to_hdf5 (file_list)
            
