
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
print(f"{len(epochs)}\n") #prints how many total events occured in the dataset


#added comme


#— — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥   
'''
def get_data(orig_data):
    my_data=orig_data[7,7,:] 
    my_data2=orig_data[7,8,:]
    my_data3=orig_data[7,9,:]
    my_data4=orig_data[7,10,:]
    
    return my_data, my_data2,my_data3,my_data4
'''


# #epochs._data[1, 2, 3]: 1- nth event out of total events, 2- channel(s) we would like to isolate, 3- datapoints we would like to include (in this case, a total of 2000 for each event)
my_data, my_data2,my_data3,my_data4=get_data(epochs.data)

fig,axs=plt.subplots(2,2) #"fig" is the overall figure/canvas, "axs" is a 2D Numpy array, "plt.subplots(nr, mc)" creates an n-row by m-column grid of subplots

ax=axs[0,0]
ax.plot(my_data)
ax=axs[0,1]
ax.plot(my_data2)
ax=axs[1,0]
ax.plot(my_data3)
ax=axs[1,1]
ax.plot(my_data4)


#— — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — —

#when does the first epoch/event occur? output: [501   0   1] where the 1st column is the sample when the event took place and the third column is the event id 
print(f"{epochs.events[0]}") 

#copies the array of events from epochs object into a new variable
events=epochs.events

#":" says to grab all datapoints for the events, "2" grabs only the 3rd column (event id) 
event_ids=events[:, 2]  


#Create a dictionary to group epoch indices by event ID:
#Creates a defaultdict where each key (event ID) will automatically map to an empty list when first accessed
event_to_epochs = defaultdict(list) 

for idx, event_id in enumerate(event_ids):
    event_to_epochs[event_id].append(idx)
    
#creates a new dictionary to hold the numpy arrays for each event ID    
event_array = {} 

# Print it out like a stem-and-leaf plot
print("\nStem-and-Leaf Style Plot: Event ID → Epoch Indices\n")
for event_id in sorted(event_to_epochs.keys()):
    indices = event_to_epochs[event_id]
    event_array[event_id] = np.array(indices)
   #print(f"Event ID {event_id}: {' '.join(map(str, indices))}")

#— — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — — — — — — — — — — —  ♥  — — — — — — — — — —
#print("\n")

for i in range(1,11):
    count = np.sum(event_ids == i )
    print(f"Event {i} occured {count} times.")

#prints the idex for all channel that can be addressed for use
print("\n",epochs.info['ch_names'])

#define a dictionary where event types are categorized by their markers
event_dict = {
   'A': [1,2],
   'D': [3,4,5],
   'L': [6,7],
   'S': [8],
   'T': [9, 10],
    }

 #— — — — — — — — — —  ♥  — — — — — — — — — —
stimtype = 'A' #for acceleration
event_idsclass = event_dict[stimtype]

accel_array = []

#input all the events associated with acceleration into a numpy array
for event_id in event_idsclass:
    if event_id in event_array:
        accel_array.extend(event_array[event_id])
        
#saves acceleration epochs into an array
accel_array = np.array(accel_array)       

accel_data=epochs._data 

times = epochs.times

plt.figure()


for idx in accel_array:
    plt.plot(times, accel_data[idx,0], alpha=0.3)    

plt.grid(True)






plt.show()
stop=1