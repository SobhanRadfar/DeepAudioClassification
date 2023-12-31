import os
from tqdm import tqdm
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa 

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)

def envelope(y, rate, thershhold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()

    for mean in y_mean:
        if mean > thershhold:
             mask.append(True)
        else:
            mask.append(False)
    return mask
dataset_path = 'wavfiles'
folder_names = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

data = pd.DataFrame(columns=['filename', 'label'])
for folder in folder_names:
    path_files = os.path.join(dataset_path, folder)
    #print(path_files)
    file_names = [f for f in  os.listdir(path_files) if os.path.isfile(os.path.join(path_files, f))]
    file_names = [os.path.join(folder, f) for f in file_names]


    folder_data = pd.DataFrame({'filename': file_names, 'label': [folder] * len(file_names)})
    data = pd.concat([data, folder_data], ignore_index=True)

data.set_index('filename', inplace=True)
#print(data)


for f in data.index:
    rate, signal = wavfile.read('wavfiles/' + f)
    data.at[f, 'length'] = signal.shape[0]/rate 

classes = list(np.unique(data.label))
class_dit = data.groupby(['label'])['length'].mean()
#print(class_dit)

fig, ax = plt.subplots()

ax.set_title('Class Disstribution', y=1.08)
ax.pie(class_dit, labels=class_dit.index, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
plt.show()


data.reset_index(inplace=True)


signals = {}
ftts = {}
fbanks = {}
mfccs = {}

for c in classes:
    file_path = data[data['label'] == c].iloc[0, 0]
    signal, rate = librosa.load(path='wavfiles/' + file_path, sr=441000)
    mask = envelope(signal, rate, thershhold=0.0005)
    signal = signal[mask]
    signals[c] = signal
    ftts[c] = calc_fft(signal, rate)

    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103)
    fbanks[c] = bank

    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
    mfccs[c] = mel



for f in tqdm(data.filename):
    signal, rate = librosa.load(path='wavfiles/' + f, sr=16000)
    mask = envelope(signal, rate, thershhold=0.0005)
    wavfile.write(filename='clean/'+f, rate=rate, data=signal[mask])

    