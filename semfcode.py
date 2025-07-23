
#— — — — — — — — — —  ♥  — — — — — — — — — —

from re import L
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import scipy as sci
import seaborn as sns
from scipy.fft import rfft, rfftfreq          # For Fast Fourier Transform (FFT)
from scipy.signal import stft, welch            # For Short-Time Fourier Transform and Power Spectral Density

#collections is a python module that specializes in container data types(dict, list, etc.), defaultdict is a more powerful dictionary that creates a new key with a default value when a user tries to access a key that doesn't exist
from collections import defaultdict

from scipy.linalg.blas import ddot 
import csv 

 #— — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —


eeg_data = mne.read_epochs(fr"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\eeg sample\cleaned_eeg\1_eeg.fif")
gsr_data = mne.read_epochs(fr"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\gsr sample\cleaned_gsr\1_gsr.fif")
emg_data = mne.read_epochs(fr"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\emg sample\cleaned_emg\1_emg.fif")
ecg_data = mne.read_epochs(fr"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\ecg sample\cleaned_ecg\1_ecg.fif")

eeg = eeg_data.get_data()
gsr = gsr_data.get_data()
emg = emg_data.get_data()
ecg = ecg_data.get_data()

# .to_data_frame() returns metadata as dataframe
eeg_m = eeg_data.to_data_frame()
print(eeg_m['time'].unique()[:100]) #prints the first 100 epochs that are unique
print(eeg_m['condition'].unique()[:5])#prints all the unique event IDs that exist
print(eeg_m['condition'].nunique()) #prints the number of unique event IDs that  exist


print(eeg.shape)
print(gsr.shape)
print(emg.shape)
print(ecg.shape)

eeg_avg = np.average(eeg, axis=(1,2)) 
gsr_avg = np.average(gsr, axis=(1,2))
emg_avg = np.average(emg, axis=(1,2))
ecg_avg = np.average(ecg, axis=(1,2))

eeg_n = eeg_m.to_numpy()
eeg_ch_mean = np.mean(eeg_n[:, 3:], axis = 1 )
eeg_ch_mean = np.mean(eeg_ch_mean.reshape(-1, 2000), axis = 1)
print(eeg_ch_mean.shape)

eeg_epochs = eeg_n[:,2]
eeg_epochs = eeg_epochs[::2000]
print(eeg_epochs.shape)

eeg_eventId = eeg_n[:,1]
eeg_eventId = eeg_eventId[::2000]
print(eeg_eventId.shape)

gsr_m = gsr_data.to_data_frame()
gsr_n = gsr_m.to_numpy()
gsr_ch_mean = gsr_n[:, 3]
print(gsr_ch_mean.shape)
gsr_ch_mean = np.mean(gsr_ch_mean.reshape(-1, 2000), axis = 1)
print(gsr_ch_mean.shape)

gsr_epochs = gsr_n[:, 2]
gsr_epochs = gsr_epochs[::2000]
print(gsr_epochs.shape)

gsr_eventId = gsr_n[:,1]
gsr_eventId = gsr_eventId[::2000]
print(gsr_eventId.shape)

emg_m = emg_data.to_data_frame()
emg_n = emg_m.to_numpy()
emg_ch_mean = np.mean(emg_n[:, 3:], axis = 1)
emg_ch_mean = np.mean(emg_ch_mean.reshape(-1, 2000), axis = 1)
print(emg_ch_mean.shape)

emg_epochs = emg_n[:, 2]
emg_epochs = emg_epochs[::2000]
print(emg_epochs.shape)

emg_eventId = emg_n[:, 1]
emg_eventId = emg_eventId[::2000]
print(emg_eventId.shape)

ecg_m = ecg_data.to_data_frame()
ecg_n = ecg_m.to_numpy()
ecg_ch_mean = ecg_n[:, 3]
ecg_ch_mean = np.mean(ecg_ch_mean.reshape(-1, 2000), axis = 1)
print(ecg_ch_mean.shape)

ecg_epochs = ecg_n[:, 3]
ecg_epochs = ecg_epochs[::2000]
print(ecg_epochs.shape)

ecg_eventId = ecg_n[:, 1]
ecg_eventId = ecg_eventId[::2000]
print(ecg_eventId.shape)

eeg_gsr_epoch = np.array_equal(eeg_epochs, gsr_epochs)
eeg_gsr_eventId = np.array_equal(eeg_eventId, gsr_eventId)


eeg_emg_epoch = np.array_equiv(eeg_epochs, emg_epochs)
eeg_emg_eventId = np.array_equiv(eeg_eventId, emg_eventId)


eeg_ecg_epoch = np.array_equiv(eeg_epochs, ecg_epochs)
eeg_ecg_eventId = np.array_equiv(eeg_eventId, ecg_eventId)


diff1 = eeg_epochs == gsr_epochs
diff2 = eeg_epochs == emg_epochs
diff3 = eeg_epochs == ecg_epochs
print(diff1)
print(diff2)
print(diff3)

avg_array = np.column_stack((eeg_epochs, gsr_epochs))
avg_array = np.column_stack((avg_array, emg_epochs))
avg_array = np.column_stack((avg_array, eeg_eventId))
avg_array = np.column_stack((avg_array, eeg_ch_mean))
avg_array = np.column_stack((avg_array, gsr_ch_mean))
avg_array = np.column_stack((avg_array, emg_ch_mean))

avg_array = pd.DataFrame(avg_array, columns=['EEG epochs', 'GSR epochs', 'EMG epochs', 'EEG event ID', 'EEG ch mean', 'GSR ch mean', 'EMG ch mean'])
avg_array = avg_array.drop(avg_array[(avg_array['EEG epochs'] != avg_array['GSR epochs']) | (avg_array['EEG epochs'] != avg_array['EMG epochs'])].index)
avg_array = avg_array.drop(['GSR epochs', 'EMG epochs'], axis = 1)

avg_array.to_csv(r"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\dataframes\avg_array.csv", index = False)


stop=1




# #— — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — — 

# epochs=mne.io.read_epochs_eeglab(fr"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\semfdata\preprocs\preprocessed data\ECG\ECG_1.set")
# print(f"{len(epochs)}\n") #prints how many total events occured in the dataset


# #added comme


# #— — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥   


# # my_data=epochs._data[7,7,:] 
# # my_data2=epochs._data[7,8,:]
# # my_data3=epochs._data[7,9,:]
# # my_data4=epochs._data[7,10,:]
    




# # # #epochs._data[1, 2, 3]: 1- nth event out of total events, 2- channel(s) we would like to isolate, 3- datapoints we would like to include (in this case, a total of 2000 for each event)


# # fig,axs=plt.subplots(2,2) #"fig" is the overall figure/canvas, "axs" is a 2D Numpy array, "plt.subplots(nr, mc)" creates an n-row by m-column grid of subplots

# # ax=axs[0,0]
# # ax.plot(my_data)
# # ax=axs[0,1]
# # ax.plot(my_data2)
# # ax=axs[1,0]
# # ax.plot(my_data3)
# # ax=axs[1,1]
# # ax.plot(my_data4)


# #— — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — —

# #when does the first epoch/event occur? output: [501   0   1] where the 1st column is the sample when the event took place and the third column is the event id 
# print(f"{epochs.events[0]}") 

# #copies the array of events from epochs object into a new variable
# events=epochs.events

# #":" says to grab all datapoints for the events, "2" grabs only the 3rd column (event id) 
# event_ids=events[:, 2]  


# #Create a dictionary to group epoch indices by event ID:
# #Creates a defaultdict where each key (event ID) will automatically map to an empty list when first accessed
# event_to_epochs = defaultdict(list) 

# for idx, event_id in enumerate(event_ids):
#     event_to_epochs[event_id].append(idx)
    
# #creates a new dictionary to hold the numpy arrays for each event ID    
# event_array = {} 

# # Print it out like a stem-and-leaf plot
# print("\nStem-and-Leaf Style Plot: Event ID → Epoch Indices\n")
# for event_id in sorted(event_to_epochs.keys()):
#     indices = event_to_epochs[event_id]
#     event_array[event_id] = np.array(indices)
#    #print(f"Event ID {event_id}: {' '.join(map(str, indices))}")

# #— — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — —
# #print("\n")

# for i in range(1,11):
#     count = np.sum(event_ids == i )
#     print(f"Event {i} occured {count} times.")

# #prints the idex for all channel that can be addressed for use
# print("\n",epochs.info['ch_names'])

# #define a dictionary where event types are categorized by their markers
# event_dict = {
#    'A': [1,2],
#    'D': [3,4,5],
#    'L': [6,7],
#    'S': [8],
#    'T': [9, 10],
#     }

#  #— — — — — — — — — —  ♥  — — — — — — — — — —
# stimtype = 'S' #for acceleration
# event_idsclass = event_dict[stimtype]

# accel_array = []

# #input all the events associated with acceleration into a numpy array
# for event_id in event_idsclass:
#     if event_id in event_array:
#         accel_array.extend(event_array[event_id])
        
# #saves acceleration epochs into an array
# accel_array = np.array(accel_array)       
# print(len(accel_array))
# accel_data=epochs._data 

# times = epochs.times

# plt.figure()


# for idx in accel_array:
#     plt.plot(times, accel_data[idx,0], alpha=0.3)    

# plt.grid(True)






# plt.show()
