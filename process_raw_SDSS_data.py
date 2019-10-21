# Created by Dmitrii Gudin (U Notre Dame).

from __future__ import division
import numpy as np
import h5py
import os
import time
import params
from scipy import interpolate
from cont_norm.spectrum_mod import Spectrum


if __name__ == '__main__':
    # Load the data files.
    File_in_params = h5py.File("download_convert_scripts/convert/params.hdf5", 'r')
    File_in_spectra = h5py.File("Data/SDSS_raw_data.hdf5", 'r')
    # Create the output file; remove the old one if exists.
    if os.path.isfile("Data/SDSS_processed_data.hdf5"):
        os.remove("Data/SDSS_processed_data.hdf5")
    File_out = h5py.File("Data/SDSS_processed_data.hdf5", 'w')        
    # Create the dataset. Leave room only for the wavelength set used in parameter estimation.
    File_out.create_dataset("/name", (len(File_in_spectra["/name"][:]), ), dtype='S14')
    File_out.create_dataset("/spectrum", (len(File_in_spectra["/spectrum"][:]), len(params.train_wavelength_list)), dtype='f')
    File_out.create_dataset("/spectrum_err", (len(File_in_spectra["/spectrum_err"][:]), len(params.train_wavelength_list)), dtype='f')
    File_out.create_dataset("/T_EFF", (len(File_in_spectra["/spectrum_err"][:]), 1), dtype='f')
    File_out.create_dataset("/T_EFF_err", (len(File_in_spectra["/spectrum_err"][:]), 1), dtype='f')
    File_out.create_dataset("/LOG_G", (len(File_in_spectra["/spectrum_err"][:]), 1), dtype='f')
    File_out.create_dataset("/LOG_G_err", (len(File_in_spectra["/spectrum_err"][:]), 1), dtype='f')
    File_out.create_dataset("/FE_H", (len(File_in_spectra["/spectrum_err"][:]), 1), dtype='f')
    File_out.create_dataset("/FE_H_err", (len(File_in_spectra["/spectrum_err"][:]), 1), dtype='f')
    File_out.create_dataset("/C_FE", (len(File_in_spectra["/spectrum_err"][:]), 1), dtype='f')
    File_out.create_dataset("/C_FE_err", (len(File_in_spectra["/spectrum_err"][:]), 1), dtype='f')
    # Perform spectral interpolation on the grid wavelength set and get the interpolated values.
    norm_spectra_list, norm_spectra_err_list = [], []
    for w, s, s_err in zip(File_in_spectra["/wavelength"][:], File_in_spectra["/spectrum"][:], File_in_spectra["/spectrum_err"][:]):
        w_1, s_1, s_err_1 = w[~np.isnan(w)], s[~np.isnan(s)], s_err[~np.isnan(s)]
        t, c, k = interpolate.splrep(w_1, s_1, k=3)
        t_err, c_err, k_err = interpolate.splrep(w_1, s_err_1, k=3)
        spline = interpolate.BSpline(t,c,k)
        spline_err = interpolate.BSpline(t_err,c_err,k_err)
        # Fill the spectra and spectral error lists with interpolated flux arrays.
        norm_spectra_list.append(spline(params.train_wavelength_list))
        norm_spectra_err_list.append(spline_err(params.train_wavelength_list))
    norm_spectra_list = np.array(norm_spectra_list)
    norm_spectra_err_list = np.array(norm_spectra_err_list)
    # Continuum-normalize the spectrum.
    for i in range(len(norm_spectra_list)):
        s = Spectrum(params.train_wavelength_list, norm_spectra_list[i])  
        continuum = s.get_continuum()
        norm_spectra_list[i]=np.divide(norm_spectra_list[i], continuum)
        norm_spectra_err_list[i] = np.divide(norm_spectra_err_list[i], continuum)  
    # Record the processed spectra.
    File_out["/name"][:] = File_in_spectra["/name"][:]
    File_out["/spectrum"][:] = norm_spectra_list
    File_out["/spectrum_err"][:] = norm_spectra_err_list
    # Fill all parameters with nans first.
    File_out["/T_EFF"][:] = np.full ((len(File_out["/T_EFF"][:]), 1), np.nan)
    File_out["/T_EFF_err"][:] = np.full ((len(File_out["/T_EFF_err"][:]), 1), np.nan)
    File_out["/LOG_G"][:] = np.full ((len(File_out["/LOG_G"][:]), 1), np.nan)
    File_out["/LOG_G_err"][:] = np.full ((len(File_out["/LOG_G_err"][:]), 1), np.nan)
    File_out["/FE_H"][:] = np.full ((len(File_out["/FE_H"][:]), 1), np.nan)
    File_out["/FE_H_err"][:] = np.full ((len(File_out["/FE_H_err"][:]), 1), np.nan)
    File_out["/C_FE"][:] = np.full ((len(File_out["/C_FE"][:]), 1), np.nan)
    File_out["/C_FE_err"][:] = np.full ((len(File_out["/C_FE_err"][:]), 1), np.nan)
    # Fill all parameters that can be found and those that make physical sense (i.e. exclude the -9999 values).
    for i in range(len(File_in_spectra["/name"][:])):
        if len(np.where(File_in_params["/name"][:]==File_out["/name"][i])[0]) > 0:
            name_index = np.where(File_in_params["/name"][:]==File_out["/name"][i])[0][0]
            if File_in_params["/T_EFF"][name_index] > 0:
                File_out["T_EFF"][i] = File_in_params["/T_EFF"][name_index]
            if File_in_params["/T_EFF_err"][name_index] > 0:
                File_out["T_EFF_err"][i] = File_in_params["/T_EFF_err"][name_index]
            if File_in_params["/LOG_G"][name_index] > -10:
                File_out["LOG_G"][i] = File_in_params["/LOG_G"][name_index]
            if File_in_params["/LOG_G_err"][name_index] > 0:
                File_out["LOG_G_err"][i] = File_in_params["/LOG_G_err"][name_index]
            if File_in_params["/FE_H"][name_index] > -10:
                File_out["FE_H"][i] = File_in_params["/FE_H"][name_index]
            if File_in_params["/FE_H_err"][name_index] > 0:
                File_out["FE_H_err"][i] = File_in_params["/FE_H_err"][name_index]
            if File_in_params["/C_FE"][name_index] > -10:
                File_out["C_FE"][i] = File_in_params["/C_FE"][name_index]
            if File_in_params["/C_FE_err"][name_index] > 0:
                File_out["C_FE_err"][i] = File_in_params["/C_FE_err"][name_index]
    # Close the files.
    File_in_spectra.close
    File_in_params.close
    File_out.close

