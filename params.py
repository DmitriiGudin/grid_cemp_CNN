# Created by Dmitrii Gudin (U Notre Dame).

from __future__ import division
import numpy as np

# FILE PARAMETERS
file_dir = "/afs/crc.nd.edu/user/d/dgudin/starnet/data/grid_cemp_R2000_clean" # Path containing T#### directories.
hdf5_file_raw = "Data/spectra_raw.hdf5" # The raw data file location.
hdf5_file_processed = "Data/spectra_processed.hdf5" # The processed (noise + normalization + wavelength cut) file location.
std_file = "Data/std.py" # The file containing STD and means for normalization.
batch_model_file = "Data/CNN_batch.hdf5" # Where to save/obtain the neural network batch model.
single_model_file = "Data/CNN_single.hdf5" # Where to save/obtain the neural network single prediction model.
shuffle_flag = 1 # Whether to shuffle dataset when converting it to hdf5. 1 if yes, else if no.
plot_folder = "Plots/" # Folder containing the output plots.


# DATA PARAMETERS
wavelength_list = range (3000, 10000+1) # List of wavelength values in the data.
train_wavelength_list = range (3817, 9198+1) # List of wavelength values used in training.
test_data_frac = 0.1 # What fraction of the whole data to use for testing. The rest is used for training and cross-validation.
cv_data_frac = 0.05 # What fraction of the training data to use for cross-validation. The rest is used for training.
S_N = 20 # Signal-to-noise ratio for noise injection.
good_percentile = 0.05 # Percentage of the spectrum within which bad spikes are expected to be.
good_frac = 10 # Maximum of all but top good_percentile flux values should not be smaller than the overall maximum by more than this factor. 
good_max_norm = 2 # Continuum normalization is considered failed and the spectrum discarded if this threshold is exceeded for any flux value.


# NEURAL NETWORK TRAINING PARAMETERS
N_spectra = 0 # Number of spectra to train/test the model on. Non-positive means working with the whole dataset. Mostly for short testing purposes.
activation = 'relu' # Activation function type for hidden layers.
initializer = 'he_normal' # How to calculate the initial model weights.
num_hidden_layers = [256, 128] # Number of neurons in hidden layers in the CNN.
num_filters = [4, 16] # Number of filters in colvolutional layers in the CNN.
filter_length = [8, 8] # Length of filters.
maxpooling_length = 4 # Length of the maxplooling window.
batch_size = 64 # Batch size (number of spectra) fed into the model per step.
max_epochs = 100 # Number of training epochs (loops over the entire data).
lr, beta_1, beta_2, optimizer_epsilon = 0.0007, 0.9, 0.999, 1e-08 # Adam optimizer parameters.
early_stopping_min_delta, early_stopping_patience, reduce_lr_factor, reduce_lr_epsilon, reduce_lr_patience, reduce_lr_min, loss_function, metrics = 0.0001, 4, 0.5, 0.0009, 2, 0.00008, 'mean_squared_error', ['accuracy', 'mae'] # Loss function parameters.
