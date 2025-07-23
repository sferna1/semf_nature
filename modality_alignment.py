import mne
import matplotlib.pyplot as plt
import numpy as np
import os

#— — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — — 

eeg_path = fr"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\ecg sample\cleaned_ecg\1_ecg.fif"
ecg_path = fr"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\semfdata\preprocs\preprocessed data\ECG\ECG_1.set"


print("Loading EEG epochs...")
eeg_epochs = mne.read_epochs(eeg_path)  # No preload
eeg_events = eeg_epochs.events
eeg_event_ids = eeg_events[:, 2]
eeg_epoch_indices = np.arange(len(eeg_event_ids))

# === LOAD ECG EPOCHS ===
print("Loading ECG epochs...")
ecg_epochs = mne.read_epochs_eeglab(ecg_path)  # No preload
ecg_events = ecg_epochs.events
ecg_event_ids = ecg_events[:, 2]
ecg_epoch_indices = np.arange(len(ecg_event_ids))

# === PLOT BOTH EVENT ID SCATTER PLOTS ===
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

# EEG plot
axs[0].scatter(eeg_epoch_indices, eeg_event_ids, marker='o', color='blue', alpha=0.7)
axs[0].set_title("EEG Event Type by Epoch (Excluding #52)")
axs[0].set_xlabel("Epoch Index")
axs[0].set_xlim(0,200)
axs[0].set_ylabel("EEG Event ID")
axs[0].set_yticks(sorted(set(eeg_event_ids)))
axs[0].grid(True)

# ECG plot
axs[1].scatter(ecg_epoch_indices, ecg_event_ids, marker='o', color='green', alpha=0.7)
axs[1].set_title("ECG Event Type by Epoch (Excluding #52)")
axs[1].set_xlabel("Epoch Index")
axs[0].set_xlim(0,200)
axs[1].set_ylabel("ECG Event ID")
axs[1].set_yticks(sorted(set(ecg_event_ids)))
axs[1].grid(True)

plt.tight_layout()
plt.show()
stop = 1
