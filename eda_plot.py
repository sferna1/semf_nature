import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#to read a .csv
df = pd.read_csv(r"C:\Users\sferna21\OneDrive - Kennesaw State University\Desktop\SEMF Project\dataframes\avg_array.csv")

#1. make figure
fig, axes = plt.subplots(3, 1)
sns.kdeplot(data=df, x = "EEG ch mean", ax = axes[0])
sns.kdeplot(data=df, x = "EMG ch mean", ax = axes[1])
sns.kdeplot(data=df, x = "GSR ch mean", ax = axes[2])
axes[0].set_title("EEG ch mean dist.") 
axes[1].set_title("EMG ch mean dist.") 
axes[2].set_title("GSR ch mean dist.") 

sns.displot(data=df, x = "EEG ch mean", ax = axes[0], kind = 'hist')
sns.displot(data=df, x = "EMG ch mean", ax = axes[1], kind = 'hist')
sns.displot(data=df, x = "GSR ch mean", ax = axes[2], kind = 'hist')
axes[0].set_title("EEG ch mean dist.") 
axes[1].set_title("EMG ch mean dist.") 
axes[2].set_title("GSR ch mean dist.") 







plt.show()