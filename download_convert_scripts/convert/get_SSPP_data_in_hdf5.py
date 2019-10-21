# Created by Dmitrii Gudin (U Notre Dame).

from __future__ import division
import numpy as np
import h5py
import os
import time
import pyfits


log_freq = 10 # How often (in terms of the number of files processed) to print logging information. Non-positive value - no logging.


def get_column(N, Type, filename): # Retrieves the data from the Nth column of a file.
    return np.genfromtxt(filename, dtype=Type, usecols=N, comments=None)


def convert_to_hdf5 (file_folder, file_list, output_file):
    print "Files to convert:", str(len(file_list))
    # Create the h5py output file, removing the old one if exists. 
    if os.path.isfile(output_file):
        os.remove(output_file)
    hdf5_file = h5py.File(output_file, 'w')
    # Define the number of spectral points.
    spectra_N = 5000
    # Create categories for all data elements.
    hdf5_file.create_dataset("/wavelength", (len(file_list), spectra_N), dtype='f')
    hdf5_file.create_dataset("/spectrum", (len(file_list), spectra_N), dtype='f')
    hdf5_file.create_dataset("/spectrum_err", (len(file_list), spectra_N), dtype='f')
    hdf5_file.create_dataset("name", (len(file_list), ), dtype='S14')
    # Fill everything with nans first.
    hdf5_file["/wavelength"][:] = np.full ((len(file_list), spectra_N), np.nan)
    hdf5_file["/spectrum"][:] = np.full ((len(file_list), spectra_N), np.nan) 
    hdf5_file["/spectrum_err"][:] = np.full ((len(file_list), spectra_N), np.nan)
    # Start time count for logging.
    time_begin = time.time()
    # Loop over all files and convert the data.
    hdf5_file["/name"][:] = np.array([f[7:21] for f in file_list])
    for f, n in zip (file_list, range(len(file_list))):
        fit = pyfits.open(file_folder+f, ignore_missing_end=True)
        hdf5_file["/wavelength"][n, range(len(fit[0].data[0].flatten()))] = 10** (fit[0].header['CRVAL1'] + np.arange(0, fit[0].header['NAXIS1'])*fit[0].header['CD1_1']).astype(float)[:]
        hdf5_file["/spectrum"][n, range(len(fit[0].data[0].flatten()))] = fit[0].data[0].flatten().astype(float)[:]
        hdf5_file["/spectrum_err"][n, range(len(fit[0].data[1].flatten()))] = fit[0].data[1].flatten().astype(float)[:]
        if log_freq>0:
            if n % log_freq == 0:
                print str(int(time.time()-time_begin)), "s: Building the hdf5 file.", str(n), "files out of", str(len(file_list)), "processed."
    # Close the file.
    hdf5_file.close()
    print str(int(time.time()-time_begin)), "s: Done."
        

if __name__ == '__main__':
    # Retrieve the file list.
    file_folder = "/scratch365/dgudin/SDSS_spectra/"
    file_list = get_column (0, str, file_folder+"file_list.txt")
    output_file = "SDSS_raw_data.hdf5"
    convert_to_hdf5 (file_folder, file_list, output_file)
            
