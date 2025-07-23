import mne
import os
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#--Configuration
labels = ['brake', 'turn', 'change', 'throttle']
event_map = {
    1: ['139', '141', '145'],
    2: ['125', '127'],
    3: ['129', '131'],
    4: ['137', '143'],
    5: ['133']
}
base_path = fr'C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\ecg sample'
#--Loop through the dataset
for dataset_num in range(1, 31):
    set_path = os.path.join(base_path, f"ECG_{dataset_num}.set")
    epochs = mne.read_epochs_eeglab(set_path)
    if '133' in epochs.event_id:
        epochs.drop([i for i, e in enumerate(epochs.events) if e[2] == epochs.event_id['133']])
    save_dir = os.path.join(base_path, "cleaned_ecg")
    os.makedirs(save_dir, exist_ok=True)
    save_name = f'{dataset_num}_ecg.fif'
    epochs.save(os.path.join(save_dir, save_name), overwrite=True)


stop = 1