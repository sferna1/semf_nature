
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

#— — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — — 

epochs=mne.io.read_epochs_eeglab(fr"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\semfdata\preprocs\preprocessed data\EEG\EEG_1.set")


data1 = epochs.get_data()
data1_avg = np.mean(data1, axis=(1, 2))
data1_reshaped = data1_avg.reshape(data1_avg.shape[0], 2)
stop = 1