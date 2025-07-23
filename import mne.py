import mne

# Paths to your .set files

raw = mne.io.read_raw_eeglab(fr'C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\eeg sample\1_brake.set')
raw.filter(l_freq=0.5, h_freq=40.)
for i, (onset, duration, description) in enumerate(zip(raw.annotations.onset, 
                                                       raw.annotations.duration, 
                                                       raw.annotations.description)):
    print(f"{i:>3}: Onset = {onset:.3f}s, Duration = {duration:.3f}s, Description = '{description}'")

stop=1