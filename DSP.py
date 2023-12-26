import os
from tqdm import tqdm
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa 


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