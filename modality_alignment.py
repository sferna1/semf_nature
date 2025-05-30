import mne
import matplotlib.pyplot as plt
import numpy as np
import os

#— — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — — 

eeg_path = fr"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\semfdata\preprocs\preprocessed data\EEG\EEG_1.set"
ecg_path = fr"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\semfdata\preprocs\preprocessed data\ECG\ECG_1.set"


print("Loading EEG dataset...")
epochs_eeg = mne.io.read_epochs_eeglab(eeg_path)
epochs_ecg =mne.io.read_epochs_eeglab(ecg_path)
print(f"Total eeg epochs: {len(epochs_eeg)}")
print(f"Total ecg epochs: {len(epochs_ecg)}")
print(f"Channel names: {epochs_eeg.info['ch_names']}")

#— — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — — 
events_eeg = epochs_eeg.events             # shape (n_events, 3)
event_eeg_ids = events[:, 2]          # Actual event IDs (e.g., 129, 135, ...)
epoch_eeg_indices = np.arange(len(event_eeg_ids))  # Epoch numbers: 0, 1, ..., N-1
 
# === PLOT: Epoch Index vs Actual Event ID ===
plt.figure(figsize=(10, 6))
plt.scatter(epoch_eeg_indices, event_eeg_ids, marker='o', color='blue', alpha=0.7)
 
plt.xlabel("Epoch Index")
plt.ylabel("Event ID")
plt.title("Event Type by Epoch")
plt.grid(True)
 
# Use actual event IDs as y-axis ticks
unique_eeg_event_ids = sorted(set(event_eeg_ids))
plt.yticks(unique_eeg_event_ids)

plt.tight_layout()
plt.show()
stop = 1
