# Created by Dmitrii Gudin (U Notre Dame).

from __future__ import division
import numpy as np
import h5py
import keras
import params
import os
import sys
import importlib
sys.path.append("/scratch365/dgudin/starnet_raw_data_SN/std")


SDSS_input_file = "Data/survey_data_processed.hdf5"
SDSS_output_file = "Output/survey_params.hdf5"
SN_list = np.arange(5, 85+1, 1)
CNN_mask = ["/scratch365/dgudin/starnet_raw_data_SN/CNN_single/CNN_single_SN_", ".hdf5"]


if __name__ == '__main__':
    # Load the SDSS data file.
    SDSS_file = h5py.File(SDSS_input_file, "r")
    N_stars = len(SDSS_file["/spectrum"][:])
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
    # Fill in the SN values.
    output_file["/S_N"][:] = np.median(np.divide(SDSS_file["/spectrum"][:], SDSS_file["/spectrum_err"][:]), axis=1)
    SN = output_file["/S_N"][:]
    # Find the closest SN values of the trained CNNs for the data sample.
    SN_pairs_min, SN_pairs_max = [], []
    for sn in SN:
        if sn<=min(SN_list):
            SN_pairs_min.append(min(SN_list))
            SN_pairs_max.append(min(SN_list))
        elif sn>=max(SN_list):
            SN_pairs_min.append(max(SN_list))
            SN_pairs_max.append(max(SN_list))
        elif sn in SN_list:
            SN_pairs_min.append(int(sn))
            SN_pairs_max.append(int(sn))
        else:
            SN_pairs_min.append (max(SN_list[SN_list<sn]))
            SN_pairs_max.append(min(SN_list[SN_list>sn]))
    # Retrieve the normalized SDSS spectra.
    data_x_min, data_x_max, data_y_min, data_y_max = [], [], [], []
    for i, sn, sn_min, sn_max in zip(range(len(SN)), SN, SN_pairs_min, SN_pairs_max):
        std_min = importlib.import_module("std_SN_"+str(sn_min))
        std_max = importlib.import_module("std_SN_"+str(sn_max))
        data_x_min.append(np.divide(SDSS_file["/spectrum"][i]-std_min.spectrum_mean,std_min.spectrum_std))
        data_y_min.append([(SDSS_file['T_EFF'][i]-std_min.T_EFF_mean)/std_min.T_EFF_std,(SDSS_file['LOG_G'][i]-std_min.LOG_G_mean)/std_min.LOG_G_std,(SDSS_file['FE_H'][i]-std_min.FE_H_mean)/std_min.FE_H_std,(SDSS_file['C_FE'][i]-std_min.C_FE_mean)/std_min.C_FE_std])
        data_x_max.append(np.divide(SDSS_file["/spectrum"][i]-std_max.spectrum_mean,std_max.spectrum_std))
        data_y_max.append([(SDSS_file['T_EFF'][i]-std_max.T_EFF_mean)/std_max.T_EFF_std,(SDSS_file['LOG_G'][i]-std_max.LOG_G_mean)/std_max.LOG_G_std,(SDSS_file['FE_H'][i]-std_max.FE_H_mean)/std_max.FE_H_std,(SDSS_file['C_FE'][i]-std_max.C_FE_mean)/std_max.C_FE_std])
    data_x_min, data_x_max = np.array(data_x_min), np.array(data_x_max)
    data_y_min, data_y_max = np.transpose(np.array(data_y_min)), np.transpose(np.array(data_y_max))
    # Get arrays of predictions.
    predict_y_min, predict_y_max, predict_y = [], [], []
    for x_min, x_max, sn, sn_min, sn_max in zip(data_x_min, data_x_max, SN, SN_pairs_min, SN_pairs_max):
        model_min = keras.models.load_model(CNN_mask[0]+str(sn_min)+CNN_mask[1])
        model_max = keras.models.load_model(CNN_mask[0]+str(sn_max)+CNN_mask[1])
        predict_y_min.append(model_min.predict(x_min.reshape(1,len(x_min),1)).flatten())
        predict_y_max.append(model_max.predict(x_max.reshape(1,len(x_max),1)).flatten())
        predict_y.append(model_min.predict(x_min.reshape(1,len(x_min),1)).flatten())
    predict_y_min = np.array(predict_y_min)
    predict_y_max = np.array(predict_y_max)
    predict_y = np.array(predict_y)
    # Convert the predictions into physical values.    
    for i in range (0, len(SN)):
        if sn_min[i]==sn_max[i]:
            std = importlib.import_module("std_SN_"+str(sn_min))        
            predict_y[i][0] = predict_y_min[i][0]*std.T_EFF_std + std.T_EFF_mean
            predict_y[i][1] = predict_y_min[i][1]*std.LOG_G_std + std.LOG_G_mean
            predict_y[i][2] = predict_y_min[i][2]*std.FE_H_std + std.FE_H_mean
            predict_y[i][3] = predict_y_min[i][3]*std.C_FE_std + std.C_FE_mean
        else:
            std_min = importlib.import_module("std_SN_"+str(sn_min))
            std_max = importlib.import_module("std_SN_"+str(sn_max))
            predict_y[i][0] = (predict_y_min[i][0]*std_min.T_EFF_std + std_min.T_EFF_mean)*abs(sn_min[i]-SN[i]) + (predict_y_max[i][0]*std_max.T_EFF_std + std_max.T_EFF_mean)*abs(sn_max[i]-SN[i])
            predict_y[i][1] = (predict_y_min[i][1]*std_min.LOG_G_std + std_min.LOG_G_mean)*abs(sn_min[i]-SN[i]) + (predict_y_max[i][1]*std_max.LOG_G_std + std_max.LOG_G_mean)*abs(sn_max[i]-SN[i])
            predict_y[i][2] = (predict_y_min[i][2]*std_min.FE_H_std + std_min.FE_H_mean)*abs(sn_min[i]-SN[i]) + (predict_y_max[i][2]*std_max.FE_H_std + std_max.FE_H_mean)*abs(sn_max[i]-SN[i])
            predict_y[i][3] = (predict_y_min[i][3]*std_min.C_FE_std + std_min.C_FE_mean)*abs(sn_min[i]-SN[i]) + (predict_y_max[i][3]*std_max.C_FE_std + std_max.C_FE_mean)*abs(sn_max[i]-SN[i]) 
    # Record all data.
    output_file["/name"][:] = SDSS_file["/name"][:]
    output_file["/T_EFF_0"][:] = SDSS_file["/T_EFF"][:].flatten()
    output_file["/LOG_G_0"][:] = SDSS_file["/LOG_G"][:].flatten()
    output_file["/FE_H_0"][:] = SDSS_file["/FE_H"][:].flatten()
    output_file["/C_FE_0"][:] = SDSS_file["/C_FE"][:].flatten()
    output_file["/T_EFF"][:] = np.transpose(predict_y)[0].flatten()
    output_file["/LOG_G"][:] = np.transpose(predict_y)[1].flatten()
    output_file["/FE_H"][:] = np.transpose(predict_y)[2].flatten()
    output_file["/C_FE"][:] = np.transpose(predict_y)[3].flatten()
