

def get_label(epochs):
    '''
    This function uses the inverse dictionary to count the labels of data 
    samples after using mne to read the dataset files.
    '''
    true_label = []
    ivd = {v: k for k, v in epochs.event_id.items()}
    for each in epochs.events:
        if ivd[each[2]] in ['139','141','145']:#brake
            true_label.append(0)
        elif ivd[each[2]] in ['125','127']:#turn
            true_label.append(1)
        elif ivd[each[2]] in ['129','131']:#change
            true_label.append(2)
        elif ivd[each[2]] in ['137','143']:#throttle
            true_label.append(3)
        elif ivd[each[2]] in ['133']:#stable
            true_label.append(4)
    return true_label

def find_class_index(label,c):
    '''
    This function finds all subscripts with category c from the given label array.
    '''
    y = []
    for i in range(len(label)):
        if label[i] == c:
            y.append(i)
    return y

def main():

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
    from mne.io import read_epochs_eeglab

    from scipy.linalg.blas import ddot 


    DATA = np.zeros(shape = (5234,64,2000))
    y = []
    label = ['brake','turn','change','throttle','stable']
    sfreq = 1000
    pointer = 0
    for i in range(2,3):
        EEG_temp = read_epochs_eeglab(fr"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\semfdata\preprocs\preprocessed data\EEG\EEG_{str(i)}.set",
                           eog=(),
                           uint16_codec=None,
                           verbose=None)
        EMG_temp = read_epochs_eeglab(fr"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\semfdata\preprocs\preprocessed data\EMG\EMG_{str(i)}.set",
                           eog=(),
                           uint16_codec=None,
                           verbose=None)
        GSR_temp = read_epochs_eeglab(fr"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\semfdata\preprocs\preprocessed data\GSR\GSR_{str(i)}.set",
                           eog=(),
                           uint16_codec=None,
                           verbose=None)
        label_EEG = get_label(EEG_temp)
        label_EMG = get_label(EMG_temp)
        label_GSR = get_label(GSR_temp)

        EEG = EEG_temp.get_data()
        EMG = EMG_temp.get_data()
        GSR = GSR_temp.get_data()

        EEG_index = []
        EMG_index = []
        GSR_index = []

        for i in range(5):
            EEG_index.append(find_class_index(label_EEG,i))
            EMG_index.append(find_class_index(label_EMG,i))
            GSR_index.append(find_class_index(label_GSR,i))

        
        # Build multimodal array according to the number of EEG
        for i in range(5):
            for j in range(min(len(EEG_index[i]),len(EMG_index[i]),len(GSR_index[i]))):
                sample = np.concatenate([EEG[EEG_index[i][j],:,:],EMG[EMG_index[i][j],:,:],GSR[GSR_index[i][j],:,:]],axis = 0)
                DATA[pointer,:,:] = sample
                pointer += 1
                y.append(i)

    print(DATA.shape)
    X_train, X_test, y_train, y_test = train_test_split(DATA, y, test_size=.2,random_state = None)
    y_train = np.array(y_train)

    print("[INFO] An overview of the data sample is as follows:")

    count_train = [0,0,0,0,0]
    for i in y_train:
        if i == 0:
            count_train[0] +=1
        elif i == 1:
            count_train[1] +=1
        elif i == 2:
            count_train[2] +=1
        elif i == 3:
            count_train[3] +=1
        elif i == 4:
            count_train[4] +=1
    print("The distribution of training data is ",count_train)
    epochs = []

    # Part II, feature extraction and classification
    samples = 2*sfreq# Number of data frame samples
    
    model = EEGNet(5, Chans = 59+5, Samples = samples, 
            dropoutRate = 0, kernLength = 512, F1 = 32,  
            D = 2, F2 = 64, norm_rate = 0.2, dropoutType = 'Dropout')
    
    lr_schedule = optimizers.schedules.learning_rate_schedule.ExponentialDecay(initial_learning_rate=0.001,
                                                                decay_steps=10000,
                                                                decay_rate=0.9)
    opt = optimizers.adam_v2.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

    print('[INFO] The network model is shown below.')
    model.summary()
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy',min_delta = 0,patience = 2500,verbose = 0,mode = 'auto',baseline = None,restore_best_weights = False)
    print('[INFO] Model training begins!')
    history = model.fit(X_train,y_train,epochs = 15000,shuffle = True,
                batch_size = 128,
                verbose = 2, 
                validation_data = tuple([np.array(X_test),np.array(y_test)]), 
                class_weight = {0:1.30,1:1.89,2:1,3:1.40,4:2.0},
                callbacks = [early_stopping])
    model.save(time.strftime("%Y%m%d%H%M%S",time.localtime())+'.h5')
    plot_result(model,history,X_test,y_test)
    print('[INFO] Finished!')

# program entrance
if __name__ == "__main__":
    main()