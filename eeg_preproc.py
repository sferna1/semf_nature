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
    4: ['137', '143']
}
eye_artifact_channels = ['VEOL', 'VEOU', 'HEOR', 'HEOL']
components = 10
threshold = 0.7
base_path = fr'C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\eeg sample'
#--Loop through the dataset
for dataset_num in range(1, 31):
    epoch_store = {} #initializes once per subject
    for event_num in range(1, 5):
        label = labels[event_num - 1]
        event_ids = event_map[event_num]
        #--Construct file path 
        filename = f"{dataset_num}_{label}.set"
        set_path = os.path.join(base_path, filename)
        if not os.path.exists(set_path):
            print(f"File not found: {set_path}")
            continue
        #--Load and flter data
        raw = mne.io.read_raw_eeglab(set_path)
        #--Add filter
        raw.filter(l_freq=0.5, h_freq=40.)

        #--Fast ICA from sklearn (algorithm) to decompose the signal into a number of components (10-20 range of components)
        data = raw.get_data()
        X = data.T #shape:(n_times, n_channels)
        ica = FastICA(n_components= components, random_state=42) 
        est = ica.fit_transform(X) #estimated components
        print(raw.ch_names)
        #find the correlation between each component and the four eog electrodes (59-63) and average them out (only one value)
        channel_indices = [raw.ch_names.index(ch) for ch in eye_artifact_channels]
        Y = raw.get_data(picks=channel_indices).T
        correlation_matrix = np.zeros((components, 4))
        for i in range(est.shape[1]):
            for j in range (Y.shape[1]):
                corr = np.corrcoef(est[:, i], Y[:, j])[0, 1]
                correlation_matrix[i, j] = corr

        #--Identify bad components
        flagged_components = (np.abs(correlation_matrix)>threshold)
        bad_ic_indices = np.where(flagged_components.any(axis=1))[0]
        print(f"Flagged ICs (|r| > {threshold}):", bad_ic_indices.tolist())

        #--Zero bad components
        est_clean= est.copy() #creates a copy of the raw w estimated components
        est_clean[:, bad_ic_indices] = 0
        X_denoised = est_clean @ ica.mixing_.T #reconstruct the cleaned signal using the ICA mixing matrix
        raw._data[:, :] = X_denoised.T 
        #--Marker extraction
        events, all_event_ids = mne.events_from_annotations(raw)
        selected_event_id = {k: v for k, v in all_event_ids.items() if k in event_ids}
        if not selected_event_id:
            print(f"No matching markers for {label} in subject {dataset_num}")
            continue
        #--Epoching 
        epochs = mne.Epochs(raw, events, event_id=selected_event_id, tmin=-0.5, tmax=1.499, baseline=(-0.5, 0))  #cannot be exactly 1.5 because mne automatically adds an extra data point for the edges
        
        epoch_store[label] = epochs
        
    #--Concatenate and save
    if len(epoch_store) == 4:
        combined_epochs = mne.concatenate_epochs([epoch_store[l] for l in labels])
        save_dir = os.path.join(base_path, "cleaned_eeg")
        os.makedirs(save_dir, exist_ok=True)
        save_name = f'{dataset_num}_eeg.fif'
        combined_epochs.save(os.path.join(save_dir, save_name), overwrite=True)



stop=1