# Created by Dmitrii Gudin (U Notre Dame).

from __future__ import division
import numpy as np
import h5py
import os
import time
import params
from cont_norm.spectrum_mod import Spectrum


# Returns True is the spectrum doesn't have false "spikes" and negative values, and False if it does and cannot be used for training.
def is_good (s):
    # Calculate the maximum flux value of the entire spectrum.
    max_val = max(s)
    # Calculate the maximum flux value of the main body of the spectrum.
    max_val_lower = max(np.sort(s)[:int((1-params.good_percentile)*len(s))])
    # Return the result of the comparison.
    return (max_val/max_val_lower<params.good_frac) and (min(s)>0)


if __name__ == '__main__':
    for i in range (5, 50):
        print "Working on S/N = ", str(i), "..."
        # Load the data file.
        File_in = h5py.File(params.hdf5_file_raw, 'r')
        # Create the output file; remove the old one if exists.
        File_out_name = "Data/spectra_processed_SN_"+str(i)+".hdf5"
        if os.path.isfile(File_out_name):
            os.remove(File_out_name)
        File_out = h5py.File(File_out_name, 'w')    
        # Perform the wavelength cut.
        wavelength_indeces = list(np.array([np.where(np.array(params.wavelength_list)==twl) for twl in params.train_wavelength_list]).flatten())
        spectrum = np.transpose(np.transpose(File_in["/spectrum"][:])[wavelength_indeces])
        # Get indeces of only good spectra.
        good_indeces_0 = []
        for i, s in enumerate(spectrum):
            if is_good(s):
                good_indeces_0.append(i)
        spectrum = spectrum[good_indeces_0]
        # Inject noise.
        spectrum = np.random.normal(spectrum, spectrum/params.S_N)
        # Continuum-normalize the noisy spectrum.
        norm_spectra_list = []
        for spec in spectrum:
            s = Spectrum(params.train_wavelength_list, spec)
            norm_spectra_list.append(s.get_flux_norm())
        # Discard spectra for which normalization failed.
        good_indeces_1 = []
        for i, s in enumerate(norm_spectra_list):
            if (0<min(s) and max(s)<params.good_max_norm):
                good_indeces_1.append(i)
        norm_spectra_list = (np.array(norm_spectra_list))[good_indeces_1]
        # Create the dataset. Leave room only for the wavelength set used in parameter estimation.
        File_out.create_dataset("/spectrum", (len(good_indeces_1), len(params.train_wavelength_list)), dtype='f')
        File_out.create_dataset("/T_EFF", (len(good_indeces_1), 1), dtype='f')
        File_out.create_dataset("/LOG_G", (len(good_indeces_1), 1), dtype='f')
        File_out.create_dataset("/FE_H", (len(good_indeces_1), 1), dtype='f')
        File_out.create_dataset("/C_FE", (len(good_indeces_1), 1), dtype='f')
        # Copy parameters.
        File_out["/T_EFF"][:] = File_in["/T_EFF"][good_indeces_0][good_indeces_1]               
        File_out["/LOG_G"][:] = File_in["/LOG_G"][good_indeces_0][good_indeces_1]
        File_out["/FE_H"][:] = File_in["/FE_H"][good_indeces_0][good_indeces_1]
        File_out["/C_FE"][:] = File_in["/C_FE"][good_indeces_0][good_indeces_1]
        # Record the processed spectra.
        File_out["/spectrum"][:] = np.array(norm_spectra_list)
        # Close the files.
        File_in.close
        File_out.close

