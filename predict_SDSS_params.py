# Created by Dmitrii Gudin (U Notre Dame).

from __future__ import division
import numpy as np
import h5py
import keras
import params
import os
import Data.std as std


SDSS_input_file = "Data/SDSS_processed_data.hdf5"
SDSS_output_file = "Output/SDSS_params.hdf5"


if __name__ == '__main__':
    # Load the SDSS data file.
    SDSS_file = h5py.File(SDSS_input_file, "r")
    N_stars = len(SDSS_file["/spectrum"][:])
    # Load the CNN single model. 
    model = keras.models.load_model(params.single_model_file)
    # Create the output file.
    if os.path.isfile(SDSS_output_file):
        os.remove(SDSS_output_file)
    output_file = h5py.File(SDSS_output_file, "w")
    # Structurize the output file.
    output_file.create_dataset("/name", (N_stars,), dtype='S14')
    output_file.create_dataset("/S_N", (N_stars,), dtype='f')
    output_file.create_dataset("/T_EFF_0", (N_stars,), dtype='f')
    output_file.create_dataset("/LOG_G_0", (N_stars,), dtype='f')
    output_file.create_dataset("/FE_H_0", (N_stars,), dtype='f')
    output_file.create_dataset("/C_FE_0", (N_stars,), dtype='f')
    output_file.create_dataset("/T_EFF", (N_stars,), dtype='f')
    output_file.create_dataset("/LOG_G", (N_stars,), dtype='f')
    output_file.create_dataset("/FE_H", (N_stars,), dtype='f')
    output_file.create_dataset("/C_FE", (N_stars,), dtype='f')
    # Retrieve the normalized SDSS spectra.
    data_x = np.divide(SDSS_file["/spectrum"][:]-std.spectrum_mean,std.spectrum_std)
    data_y = np.column_stack((((SDSS_file['T_EFF'][:]-std.T_EFF_mean)/std.T_EFF_std),((SDSS_file['LOG_G'][:]-std.LOG_G_mean)/std.LOG_G_std),((SDSS_file['FE_H'][:]-std.FE_H_mean)/std.FE_H_std),((SDSS_file['C_FE'][:]-std.C_FE_mean)/std.C_FE_std)))
    # Get an array of predictions.
    predict_y = []
    for x in data_x:
        predict_y.append(model.predict(x.reshape(1,len(x),1)).flatten())
    predict_y = np.array(predict_y)
    # Convert the predictions into physical values.    
    for i in range (0, len(data_y)):
        predict_y[i][0] = predict_y[i][0]*std.T_EFF_std + std.T_EFF_mean
        predict_y[i][1] = predict_y[i][1]*std.LOG_G_std + std.LOG_G_mean
        predict_y[i][2] = predict_y[i][2]*std.FE_H_std + std.FE_H_mean
        predict_y[i][3] = predict_y[i][3]*std.C_FE_std + std.C_FE_mean
    # Record all data.
    output_file["/name"][:] = SDSS_file["/name"][:]
    output_file["/S_N"][:] = np.median(np.divide(SDSS_file["/spectrum"][:], SDSS_file["/spectrum_err"][:]), axis=1)
    output_file["/T_EFF_0"][:] = SDSS_file["/T_EFF"][:].flatten()
    output_file["/LOG_G_0"][:] = SDSS_file["/LOG_G"][:].flatten()
    output_file["/FE_H_0"][:] = SDSS_file["/FE_H"][:].flatten()
    output_file["/C_FE_0"][:] = SDSS_file["/C_FE"][:].flatten()
    output_file["/T_EFF"][:] = np.transpose(predict_y)[0].flatten()
    output_file["/LOG_G"][:] = np.transpose(predict_y)[1].flatten()
    output_file["/FE_H"][:] = np.transpose(predict_y)[2].flatten()
    output_file["/C_FE"][:] = np.transpose(predict_y)[3].flatten()
