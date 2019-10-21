# Created by Dmitrii Gudin (U Notre Dame).

from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
import keras
import params
import Data.std as std


# Plots the residuals (individual, mean, std) for stellar parameters.
def plot_residuals (data_y, residuals):
    for stel_par in range(0,4):
        # Retrieve the specified data.
        val, res = np.transpose(data_y)[stel_par], np.transpose(residuals)[stel_par]
        # Get the set of the training parameter values.
        param_val = np.array(sorted(set(val)))
        # Calculate mean and std for residuals at these values.
        res_mean, res_std = [], []
        for p in param_val:
            res_mean.append (np.mean([r for r,v in zip(res,val) if v==p]))
            res_std.append (np.std([r for r,v in zip(res,val) if v==p]))
        res_mean = np.array(res_mean)
        res_std = np.array(res_std)
        # Make a plot.
        plt.clf()
        plt.title("Residuals", size=24)
        if stel_par==0:
            plt.xlabel("Teff", size=24) 
        elif stel_par==1:
            plt.xlabel("log(g)", size=24)
        elif stel_par==2:
            plt.xlabel("Z", size=24)
        elif stel_par==3:
            plt.xlabel("[C/Fe]", size=24)
        plt.ylabel("Residual value", size=24)
        plt.tick_params(labelsize=18)
        plt.xlim(min(param_val),max(param_val))
        plt.ylim(min(res), max(res))
        plt.plot([min(param_val),max(param_val)], [0,0], color='black', linewidth=1, linestyle='--')
        recs = [mpatches.Rectangle((0,0),1,1,fc='red'), mpatches.Rectangle((0,0),1,1,fc='blue'), mpatches.Rectangle((0,0),1,1,fc='green')]
        legend_labels = ["Residuals", "Mean", "STD"]
        plt.legend(recs, legend_labels, loc=4, fontsize=18)
        plt.scatter(val, res, color='red', s=1, marker='o')
        plt.plot(param_val, res_mean, color='blue', linewidth=3)
        plt.fill_between (param_val, res_mean-res_std, res_mean+res_std, facecolor='green', interpolate=True, alpha=0.33)
        plt.gcf().set_size_inches(25.6, 14.6)
        if stel_par==0:
            filename=params.plot_folder+'Res_Teff.png'
        elif stel_par==1:
            filename=params.plot_folder+'Res_Logg.png'
        elif stel_par==2:
            filename=params.plot_folder+'Res_Z.png'
        elif stel_par==3:
            filename=params.plot_folder+'Res_C_FE.png'
        plt.gcf().savefig(filename, dpi=100)
        plt.close()


if __name__ == '__main__':
    # Load the data file.
    File = h5py.File(params.hdf5_file_processed,'r')
    # How many testing spectra do we have?
    if not params.N_spectra<=0:
        N_spectra = min(int(len(File['spectrum'][:])*params.test_data_frac), int(params.N_spectra*params.test_data_frac))
    else: 
        N_spectra = int(len(File['spectrum'][:])*params.test_data_frac)
    # Retrieve the normalized testing spectra.
    data_x = np.divide(File['spectrum'][:]-std.spectrum_mean,std.spectrum_std)[-N_spectra:]
    data_y = np.column_stack((((File['T_EFF'][:]-std.T_EFF_mean)/std.T_EFF_std)[-N_spectra:],((File['LOG_G'][:]-std.LOG_G_mean)/std.LOG_G_std)[-N_spectra:],((File['FE_H'][:]-std.FE_H_mean)/std.FE_H_std)[-N_spectra:],((File['C_FE'][:]-std.C_FE_mean)/std.C_FE_std)[-N_spectra:]))
    # Load the CNN batch model.
    model = keras.models.load_model(params.single_model_file)
    # Get an array of predictions.
    predict_y = []
    for x in data_x:
        predict_y.append(model.predict(x.reshape(1,len(x),1)).flatten())
    predict_y = np.array(predict_y)
    # Convert the data and the predictions into physical values.    
    for i in range (0, len(data_y)):
        data_y[i][0] = data_y[i][0]*std.T_EFF_std + std.T_EFF_mean
        data_y[i][1] = data_y[i][1]*std.LOG_G_std + std.LOG_G_mean
        data_y[i][2] = data_y[i][2]*std.FE_H_std + std.FE_H_mean
        data_y[i][3] = data_y[i][3]*std.C_FE_std + std.C_FE_mean
        predict_y[i][0] = predict_y[i][0]*std.T_EFF_std + std.T_EFF_mean
        predict_y[i][1] = predict_y[i][1]*std.LOG_G_std + std.LOG_G_mean
        predict_y[i][2] = predict_y[i][2]*std.FE_H_std + std.FE_H_mean
        predict_y[i][3] = predict_y[i][3]*std.C_FE_std + std.C_FE_mean
    # Obtain the array of residuals.
    residuals = []
    for i in range (0, len(data_y)):
        residuals.append (predict_y[i]-data_y[i])
    residuals = np.array(residuals)
    # Plot the residuals.
    plot_residuals (data_y, residuals)
    # Calculate and print out mean and STD errors overall.
    print "Teff mean error: ", np.mean(np.transpose(residuals)[0])
    print "Teff error STD: ", np.std(np.transpose(residuals)[0])
    print "log(g) mean error: ", np.mean(np.transpose(residuals)[1])
    print "log(g) error STD: ", np.std(np.transpose(residuals)[1]) 
    print "Z mean error: ", np.mean(np.transpose(residuals)[2])
    print "Z error STD: ", np.std(np.transpose(residuals)[2])
    print "[C/Fe] mean error: ", np.mean(np.transpose(residuals)[3])
    print "[C/Fe] error STD: ", np.std(np.transpose(residuals)[3])
