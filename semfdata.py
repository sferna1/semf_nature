
#— — — — — — — — — —  ♥  — — — — — — — — — —

from re import L
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import scipy as sci
from scipy.fft import rfft, rfftfreq          # For Fast Fourier Transform (FFT)
from scipy.signal import stft, welch            # For Short-Time Fourier Transform and Power Spectral Density
#collections is a python module that specializes in container data types(dict, list, etc.), defaultdict is a more powerful dictionary that creates a new key with a default value when a user tries to access a key that doesn't exist
from collections import defaultdict

from scipy.linalg.blas import ddot 

#— — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — —  — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — — 
eegpath =fr"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\semfdata\preprocs\preprocessed data\EEG\EEG_1.set"
eeg_epochs=mne.io.read_epochs_eeglab(eegpath)

eeg_data1 = eeg_epochs.get_data() #get all the epochs and store it in data1
eeg_data1_avg = np.mean(eeg_data1, axis=(1, 2)) #using numpy, find the mean of each epoch, 
eeg_data1_reshaped = eeg_data1_avg.reshape(eeg_data1_avg.shape[0], 1) #uses the epochs available for each subject and turns the data's shape into a vector

ecgpath =fr"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\semfdata\preprocs\preprocessed data\ECG\ECG_1.set"
ecg_epochs=mne.io.read_epochs_eeglab(ecgpath)

ecg_data1 = ecg_epochs.get_data() #get all the epochs and store it in data1
ecg_data1_avg = np.mean(ecg_data1, axis=(1, 2)) #using numpy, find the mean of each epoch, 
ecg_data1_reshaped = ecg_data1_avg.reshape(ecg_data1_avg.shape[0], 1) #uses the epochs available for each subject and turns the data's shape into a vector

emgpath = fr"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\semfdata\preprocs\preprocessed data\EMG\EMG_1.set"
emg_epochs=mne.io.read_epochs_eeglab(emgpath)

emg_data1 = emg_epochs.get_data() #get all the epochs and store it in data1
emg_data1_avg = np.mean(emg_data1, axis=(1, 2)) #using numpy, find the mean of each epoch, 
emg_data1_reshaped = emg_data1_avg.reshape(emg_data1_avg.shape[0], 1) #uses the epochs available for each subject and turns the data's shape into a vector


stop = 1