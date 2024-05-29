# Scan the audio file directory and check file integrity
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
