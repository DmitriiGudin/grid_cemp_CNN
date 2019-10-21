# Created by Dmitrii Gudin (U Notre Dame).

from __future__ import division
import numpy as np
import h5py
import os
import random
import params

from keras.models import Sequential
from keras.layers import Dense, InputLayer, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Calculates the means and STDs for each spectrum element and the stellar parameters. Saves them to the file specified in params.py. Also returns the normalized arrays.
def save_mean_std (spectrum, T_EFF, LOG_G, FE_H, C_FE):
    # Calculate STDs and means.
    spectrum_std = np.std(spectrum, axis=0)
    T_EFF_std = np.std(T_EFF)
    LOG_G_std = np.std(LOG_G)
    FE_H_std = np.std(FE_H)
    C_FE_std = np.std(C_FE)
    spectrum_mean = np.mean(spectrum, axis=0)
    T_EFF_mean = np.mean(T_EFF)
    LOG_G_mean = np.mean(LOG_G)
    FE_H_mean = np.mean(FE_H)
    C_FE_mean = np.mean(C_FE)
    # Save them to a file.
    if os.path.isfile(params.std_file):
        os.remove(params.std_file)
    with open(params.std_file,'w') as std_file:
        std_file.write("from __future__ import division\n")
        std_file.write("import numpy as np\n")
        std_file.write("\n")
        std_file.write("\n")
        std_file.write("# This file contains the values for STD and means of each variable in the full dataset. To use CNN for testing or parameter predictions, please use these values to normalize your spectrum (make sure it has the same set of wavelengths as the one the CNN trained on!) and parameters like so:\n")
        std_file.write("#\n")
        std_file.write("# spectrum = np.divide(spectrum-std_file.spectrum_mean,std_file.spectrum_std)\n")
        std_file.write("# T_EFF = (T_EFF-std_file.T_EFF_mean)/std_file.T_EFF_std\n")
        std_file.write("# LOG_G = (LOG_G-std_file.LOG_G_mean)/std_file.LOG_G_std\n")
        std_file.write("# FE_H = (FE_H-std_file.FE_H_mean)/std_file.FE_H_std\n")
        std_file.write("# C_FE = (C_FE-std_file.C_FE_mean)/std_file.C_FE_std\n")
        std_file.write("\n")
        std_file.write("\n")
        std_file.write("spectrum_std = np.array(" + str(list(spectrum_std)) + ")\n")   
        std_file.write('T_EFF_std = %s\n' % T_EFF_std)
        std_file.write('LOG_G_std = %s\n' % LOG_G_std)
        std_file.write('FE_H_std = %s\n' % FE_H_std)
        std_file.write('C_FE_std = %s\n' % C_FE_std)
        std_file.write("spectrum_mean = np.array(" + str(list(spectrum_mean)) + ")\n") 
        std_file.write('T_EFF_mean = %s\n' % T_EFF_mean)
        std_file.write('LOG_G_mean = %s\n' % LOG_G_mean)
        std_file.write('FE_H_mean = %s\n' % FE_H_mean)
        std_file.write('C_FE_mean = %s\n' % C_FE_mean)
    # And return these values.
    return spectrum_std, T_EFF_std, LOG_G_std, FE_H_std, C_FE_std, spectrum_mean, T_EFF_mean, LOG_G_mean, FE_H_mean, C_FE_mean


# Train and cross-validation batch generator.
def batch_gen (data_x, data_y):
    while True:
        # Generate a random set of indeces.
        indeces = random.sample(range(0,len(data_x)),params.batch_size)
        # Yield the data for those indeces, reshaping it properly.
        yield (np.array([data_x[i] for i in indeces]).reshape(params.batch_size, len(data_x[0]), 1), np.array([data_y[i] for i in indeces]))


# Creates and sets up a CNN model with parameters specified in params.py.
def create_model(batch_size):
    return Sequential([InputLayer(batch_input_shape=(batch_size, len(params.train_wavelength_list), 1)),
    Conv1D(kernel_initializer=params.initializer, activation=params.activation, padding="same", filters=params.num_filters[0], kernel_size=params.filter_length[0]),
    Conv1D(kernel_initializer=params.initializer, activation=params.activation, padding="same", filters=params.num_filters[1], kernel_size=params.filter_length[1]),
    MaxPooling1D(pool_size=params.maxpooling_length),
    Flatten(),
    Dense(units=params.num_hidden_layers[0], kernel_initializer=params.initializer, activation=params.activation),
    Dense(units=params.num_hidden_layers[1], kernel_initializer=params.initializer, activation=params.activation),
    Dense(units=4, activation="linear", input_dim=params.num_hidden_layers[1]),
])


# Creates and sets up an optimizer for the CNN model with parameters specified in params.py.
def create_optimizer():
    return Adam(lr=params.lr, beta_1=params.beta_1, beta_2=params.beta_2, epsilon=params.optimizer_epsilon, decay=0.0)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=params.early_stopping_min_delta, 
    patience=params.early_stopping_patience, verbose=2, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=params.reduce_lr_epsilon, 
    patience=params.reduce_lr_patience, min_lr=params.reduce_lr_min, mode='min', verbose=2)

if __name__ == '__main__':
    # Load the data file.
    File = h5py.File(params.hdf5_file_processed, 'r')
    # Calculate the means and STDs for all variables involved. Save them to a file.
    spectrum_std, T_EFF_std, LOG_G_std, FE_H_std, C_FE_std, spectrum_mean, T_EFF_mean, LOG_G_mean, FE_H_mean, C_FE_mean = save_mean_std(File['spectrum'][:], File['T_EFF'][:], File['LOG_G'][:], File['FE_H'][:], File['C_FE'][:])
    # How many training and cross-validation spectra do we have?
    if not params.N_spectra<=0:
        N_spectra = min(int(len(File['spectrum'][:])*(1-params.test_data_frac)), params.N_spectra)
    else:
        N_spectra = int(len(File['spectrum'][:])*(1-params.test_data_frac))
    # Get the list of indices for the spectral part used for training.
    wavelength_indeces = list(np.array([np.where(np.array(params.wavelength_list)==twl) for twl in params.train_wavelength_list]).flatten())
    # Normalize the variables and prepare them for training.
    data_x = np.divide(File['spectrum'][:]-spectrum_mean,spectrum_std)[0:N_spectra]
    data_y = np.column_stack((((File['T_EFF'][:]-T_EFF_mean)/T_EFF_std)[0:N_spectra],((File['LOG_G'][:]-LOG_G_mean)/LOG_G_std)[0:N_spectra],((File['FE_H'][:]-FE_H_mean)/FE_H_std)[0:N_spectra],((File['C_FE'][:]-C_FE_mean)/C_FE_std)[0:N_spectra]))
    # How many training/cross-validation spectra do we have?
    N_train = int(N_spectra*(1-params.cv_data_frac))
    # Split the data into training/cross-validation data and clean up the obsolete data.
    data_x_train, data_y_train = data_x[0:N_train], data_y[0:N_train]
    data_x_cv, data_y_cv = data_x[N_train:N_spectra], data_y[N_train:N_spectra]
    # Generate CNN model.
    batch_model = create_model(params.batch_size)
    # Generate optimizer for the model and some Keras functions.
    optimizer = create_optimizer()
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=params.early_stopping_min_delta, patience=params.early_stopping_patience, verbose=2, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=params.reduce_lr_epsilon, patience=params.reduce_lr_patience, min_lr=params.reduce_lr_min, mode='min', verbose=2)
    # Compile the model.
    batch_model.compile(optimizer=optimizer, loss=params.loss_function, metrics=params.metrics)
    # Run the training procedure.
    batch_model.fit_generator(batch_gen(data_x_train, data_y_train), steps_per_epoch = int(N_train/params.batch_size), epochs=params.max_epochs, validation_data=batch_gen(data_x_cv, data_y_cv), max_queue_size=10, verbose=2, callbacks=[early_stopping, reduce_lr], validation_steps=int((N_spectra-N_train)/params.batch_size))
    # Save the batch model.
    batch_model.save(params.batch_model_file)
    # Save the single prediction model.
    single_model = create_model(1)
    single_model.set_weights(batch_model.get_weights())
    single_model.compile(optimizer=optimizer, loss=params.loss_function, metrics=params.metrics)
    single_model.save(params.single_model_file)
    print "Training done."
