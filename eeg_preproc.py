import mne
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
# Paths to your .set files
raw = mne.io.read_raw_eeglab(fr'C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\eeg sample\1_brake.set')
print(raw.ch_names)
#add filter
raw.filter(l_freq=0.5, h_freq=40.)
for i, (onset, duration, description) in enumerate(zip(raw.annotations.onset, 
                                                       raw.annotations.duration, 
                                                       raw.annotations.description)):
    print(f"{i:>3}: Onset = {onset:.3f}s, Duration = {duration:.3f}s, Description = '{description}'")


#fast ICA from sklearn (algorithm) to decompose the signal into a number of components (10-20 range of components)
#raw.pick_types(eeg=True) #skips EOG, ECG
data = raw.get_data()
print(data.shape)
X = raw.get_data().T #shape:(n_times, n_channels)
print(X.shape)
components = 10
raw_ica = FastICA(n_components= components, random_state=42) 
raw_est = raw_ica.fit_transform(X) #estimated components
print(raw_est.shape)

#find the correlation between each component and the four eog electrodes (59-63) and average them out (only one value)
eye_artifact_channels = ['VEOL', 'VEOU', 'HEOR', 'HEOL']
channel_indices = [raw.ch_names.index(ch) for ch in eye_artifact_channels]
Y = raw.get_data(picks=channel_indices).T
correlation_matrix = np.zeros((components, 4))
for i in range(raw_est.shape[1]):
    for j in range (Y.shape[1]):
        corr = np.corrcoef(raw_est[:, i], Y[:, j])[0, 1]
        correlation_matrix[i, j] = corr

df_corr = pd.DataFrame(correlation_matrix, columns=eye_artifact_channels, index=[f'IC{i+1}' for i in range(components)])
print(df_corr)     

#set a threshold after finding correlation values
threshold = 0.7
flagged_components = (np.abs(correlation_matrix)>threshold)
bad_ic_indices = np.where(flagged_components.any(axis=1))[0]
print(f"Flagged ICs (|r| > {threshold}):", bad_ic_indices.tolist())


plt.figure(figsize=(10, 6))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Between ICA Components and EEG Channels")
plt.tight_layout()
plt.show()
#have ICs that satisfy the above threshold eliminated
raw_est_clean= raw_est.copy()
raw_est_clean[:, bad_ic_indices] = 0
X_denoised = raw_est_clean @ raw_ica.mixing_.T #reconstruct the cleaned signal using the ICA mixing matrix
raw._data[:, :] = X_denoised.T 
#apply new ICA on your data, like an inverse transform, this yeilds new data that resembles the orginal data in shape but different in values




stop=1