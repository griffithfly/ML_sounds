## Package Dependencies
```
import os
import math, random
import wave
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, random_split, DataLoader
from scipy.io.wavfile import read as read_wav
from natsort import natsorted
import librosa
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from torchaudio import transforms
from IPython.display import Audio
```
## Scan the audio file directory and check file integrity
```
path = '/class0'
file_list = os.listdir(path)
c0_list = []
for f in file_list:
    if os.path.splitext(f)[1] == '.wav':
        c0_list.append(f)
c0_list = natsorted(c0_list)

path = '/class1'
file_list = os.listdir(path)
c1_list = []
for f in file_list:
    if os.path.splitext(f)[1] == '.wav':
        c1_list.append(f)
c1_list = natsorted(c1_list)

print(len(c0_list), len(c1_list))
```
<img src="" alt="print_size" style="height: 100px; width:100px;"/>
