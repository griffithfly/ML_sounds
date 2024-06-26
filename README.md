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
## Scan the audio file directory and check data integrity
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
50 20
# Undestanding Data
## Look a single data
```
metadata = torchaudio.info('class1/0.wav')
print(metadata)
```
AudioMetaData(sample_rate=16000, num_frames=160000, num_channels=1, bits_per_sample=64, encoding=PCM_F)
## Visualize this single data
```
waveform, sample_rate = torchaudio.load('class0/0.wav')

print_stats(waveform, sample_rate=sample_rate)
plot_waveform(waveform, sample_rate)
plot_specgram(waveform, sample_rate)
```
Sample Rate: 16000
Shape: (1, 160000)
Dtype: torch.float32
 - Max:      1.000
 - Min:     -0.307
 - Mean:     0.419
 - Std Dev:  0.275

tensor([[-0.0329, -0.0375, -0.0166,  ..., -0.2747, -0.2009, -0.2282]])

<img src="pics/1.png" style="height: 100px; width:150px;"/>
<img src="pics/2.png" style="height: 100px; width:150px;"/>

## Visualize all raw data
### Preparation
### Viusal Class0
```
def print_stats(waveform, sample_rate=None, src=None):
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  if sample_rate:
    print("Sample Rate:", sample_rate)
  print("Shape:", tuple(waveform.shape))
  print("Dtype:", waveform.dtype)
  print(f" - Max:     {waveform.max().item():6.3f}")
  print(f" - Min:     {waveform.min().item():6.3f}")
  print(f" - Mean:    {waveform.mean().item():6.3f}")
  print(f" - Std Dev: {waveform.std().item():6.3f}")
  print()
  print(waveform)
  print()
    
def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show(block=False)

def plot_waveforms(waveforms, sample_rate, titles=None, xlim=None, ylim=None):
    num_waveforms, num_channels, num_samples = waveforms.shape
    time_axis = torch.arange(0, num_samples) / sample_rate

    num_rows = min(num_waveforms, 4)
    num_cols = min((num_waveforms + num_rows - 1) // num_rows, 5)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 10))
    axes = axes.ravel() 

    for i in range(num_rows * num_cols):
        if i < num_waveforms:
            for c in range(num_channels):
                ax = axes[i * num_channels + c]  
                ax.plot(time_axis.numpy(), waveforms[i, c].numpy(), linewidth=1)
                ax.grid(True)
                if titles:
                    ax.set_title(titles[i] + f" - Channel {c+1}")
                else:
                    ax.set_title(f"Waveform {i+1} - Channel {c+1}")
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')

                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)

    plt.tight_layout()
    plt.show()

all_waveforms = []

for c0 in sorted(c0_list):
    waveform, sample_rate = torchaudio.load('class0/' + str(c0))
    all_waveforms.append(waveform)
    
all_waveforms_tensor = torch.stack(all_waveforms)
plot_waveforms(all_waveforms_tensor, sample_rate, titles=c0_list, xlim=None, ylim=None)
```
<img src="pics/3.png" style="height: 495px; width:892px;"/>

### Viusal Class1

```
all_waveforms = []
    
for c1 in sorted(c1_list):
    waveform, sample_rate = torchaudio.load('class1/' + str(c1))
    all_waveforms.append(waveform)
    
all_waveforms_tensor = torch.stack(all_waveforms)
plot_waveforms(all_waveforms_tensor, sample_rate, titles=c1_list, xlim=None, ylim=None)
```

<img src="pics/4.png" style="height: 495px; width:892px;"/>

## Determine to normalize or not
### Check sample rate to see if we need to Standardize sampling rate

```
for c0 in sorted(c0_list):
    waveform, sample_rate = torchaudio.load('class0/' + str(c1))
    print(c0, sample_rate)

for c1 in sorted(c1_list):
    waveform, sample_rate = torchaudio.load('class1/' + str(c1))
    print(c1, sample_rate)
```

## Determine Data Augmentation or not
### apply effects to data to see results

```
waveform1, sample_rate1 = torchaudio.load('class1/0.wav')

effects = [
  ["lowpass", "-1", "300"], 
  ["speed", "0.8"],  
  ["rate", f"{sample_rate1}"],
  ["reverb", "-w"],  
]

waveform2, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(
    waveform1, sample_rate1, effects)

plot_waveform(waveform1, sample_rate1, title="Original", xlim=(-.1, 3.2))
plot_waveform(waveform2, sample_rate2, title="Effects Applied", xlim=(-.1, 3.2))
print_stats(waveform1, sample_rate=sample_rate1, src="Original")
print_stats(waveform2, sample_rate=sample_rate2, src="Effects Applied")
```

<img src="pics/5.png" style="height: 100px; width:150px;"/>
<img src="pics/6.png" style="height: 100px; width:150px;"/>


==Source: Original==
Sample Rate: 16000
Shape: (1, 160000)
Dtype: torch.float32
 - Max:      0.985
 - Min:     -1.000
 - Mean:    -0.150
 - Std Dev:  0.482

tensor([[ 0.0749, -0.0468,  0.0461,  ..., -0.2309, -0.2032, -0.2890]])

==Source: Effects Applied==
Sample Rate: 16000
Shape: (2, 200000)
Dtype: torch.float32
 - Max:      0.815
 - Min:     -0.824
 - Mean:    -0.139
 - Std Dev:  0.476

tensor([[ 0.0000,  0.0000,  0.0000,  ..., -0.4799, -0.4832, -0.4840],
        [ 0.0000,  0.0000,  0.0000,  ..., -0.4758, -0.4765, -0.4737]])

## Determine Feature Augmentation or not

```
SAMPLE_WAV_SPEECH_PATH = 'class0/0.wav'

def _get_sample(path, resample=None):
  effects = [
    ["remix", "1"]
  ]
  if resample:
    effects.extend([
      ["lowpass", f"{resample // 2}"],
      ["rate", f'{resample}'],
    ])
  return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

def get_speech_sample(*, resample=None):
  return _get_sample(SAMPLE_WAV_SPEECH_PATH, resample=resample)

def get_spectrogram(
    n_fft = 400,
    win_len = None,
    hop_len = None,
    power = 2.0,
):
  waveform, _ = get_speech_sample()
  spectrogram = T.Spectrogram(
      n_fft=n_fft,
      win_length=win_len,
      hop_length=hop_len,
      center=True,
      pad_mode="reflect",
      power=power,
  )
  return spectrogram(waveform)

spec = get_spectrogram(power=None)
strech = T.TimeStretch()

rate = 1.2
spec_ = strech(spec, rate)
plot_spectrogram(spec_[0].abs(), title=f"Stretched x{rate}", aspect='equal', xmax=304)

plot_spectrogram(spec[0].abs(), title="Original", aspect='equal', xmax=304)

rate = 0.9
spec_ = strech(spec, rate)
plot_spectrogram(spec_[0].abs(), title=f"Stretched x{rate}", aspect='equal', xmax=304)
```

<img src="pics/7.png" style="height: 100px; width:150px;"/>
<img src="pics/8.png" style="height: 100px; width:150px;"/>
<img src="pics/9.png" style="height: 100px; width:150px;"/>

## Determine pitch or not
```
def plot_pitch(waveform, sample_rate, pitch, ax):
    ax.set_title("Pitch Feature")
    ax.grid(True)

    end_time = waveform.shape[1] / sample_rate
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    ax.plot(time_axis, waveform[0], linewidth=1, color='gray', alpha=0.3)

    ax2 = ax.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    ax2.plot(
        time_axis, pitch[0], linewidth=2, label='Pitch', color='green')

    ax2.legend(loc=0)

path = '/class0'

file_list = os.listdir(path)
c0_list = [os.path.join(path, f) for f in file_list if os.path.splitext(f)[1] == '.wav']
c0_list = natsorted(c0_list)

num_files = len(c0_list)
num_rows = (num_files + 4) // 5  # Round up to the nearest multiple of 5
num_cols = 5

fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 10))

for idx, file_path in enumerate(c0_list):
    row_idx = idx // num_cols
    col_idx = idx % num_cols
    ax = axes[row_idx, col_idx] if num_files > 1 else axes
    waveform, sample_rate = torchaudio.load(file_path)
    pitch = F.detect_pitch_frequency(waveform, sample_rate)
    plot_pitch(waveform, sample_rate, pitch, ax)

for idx in range(len(c0_list), num_rows * num_cols):
    row_idx = idx // num_cols
    col_idx = idx % num_cols
    ax = axes[row_idx, col_idx] if num_files > 1 else axes
    ax.axis('off')

plt.tight_layout()
plt.show()
```

<img src="pics/10.png" style="height: 600px; width:1000px;"/>

## Check waveform shape
```
path = 'class0'
file_list = os.listdir(path)
c0_list = [os.path.join(path, f) for f in file_list if os.path.splitext(f)[1] == '.wav']
c0_list = natsorted(c0_list)

path = 'class1'
file_list = os.listdir(path)
c1_list = [os.path.join(path, f) for f in file_list if os.path.splitext(f)[1] == '.wav']
c1_list = natsorted(c1_list)

audio_file = c0_list + c1_list

class AudioUtil():
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)

for audio_path in audio_file:
    aud = AudioUtil.open(audio_path)
    waveform, sample_rate = aud
    print("Waveform shape:", waveform.shape)
```

Waveform shape: torch.Size([1, 160000])

# Build CNNs
```
class SoundDS(Dataset):
    def __init__(self, audio_paths):
        self.audio_paths = audio_paths

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        # You can add more preprocessing or extract labels if available
        return waveform, sample_rate

# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        # Ensure the input has the correct shape
        if x.ndim == 3:  
            x = x.unsqueeze(1)  
        x = self.conv(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)
        return x

myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)

class SoundDS(Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        label = self.labels[idx]
        return waveform, label

path1 = '/class0'  
path2 = '/class1'  

file_list1 = os.listdir(path1)
file_list2 = os.listdir(path2)
c0_list = [os.path.join(path1, f) for f in file_list1 if os.path.splitext(f)[1] == '.wav']
c1_list = [os.path.join(path2, f) for f in file_list2 if os.path.splitext(f)[1] == '.wav']
audio_file = c0_list + c1_list
myds = SoundDS(audio_file)

labels1 = [0] * len(os.listdir(path1))  
labels2 = [1] * len(os.listdir(path2))  

num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

def training(model, train_dl, val_dl, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        for i, data in enumerate(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        print(f'Epoch: {epoch+1}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

        train_losses.append(avg_loss)
        train_accs.append(acc)

        model.eval()
        val_running_loss = 0.0
        val_correct_prediction = 0
        val_total_prediction = 0
        with torch.no_grad():
            for i, data in enumerate(val_dl):
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, prediction = torch.max(outputs, 1)
                val_correct_prediction += (prediction == labels).sum().item()
                val_total_prediction += prediction.shape[0]

        val_num_batches = len(val_dl)
        val_avg_loss = val_running_loss / val_num_batches
        val_acc = val_correct_prediction / val_total_prediction
        print(f'Validation Loss: {val_avg_loss:.2f}, Validation Accuracy: {val_acc:.2f}')

        val_losses.append(val_avg_loss)
        val_accs.append(val_acc)

    print('Finished Training')
    
    # Plotting the loss and accuracy curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accs, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

num_epochs = 10   
training(myModel, train_dl, val_dl, num_epochs)
```
- Epoch: 1, Loss: 2.31, Accuracy: 0.12
- Validation Loss: 2.30, Validation Accuracy: 0.00
- Epoch: 2, Loss: 2.20, Accuracy: 0.39
- Validation Loss: 2.28, Validation Accuracy: 0.00
- Epoch: 3, Loss: 2.05, Accuracy: 0.80
- Validation Loss: 2.19, Validation Accuracy: 0.64
- Epoch: 4, Loss: 1.79, Accuracy: 1.00
- Validation Loss: 2.03, Validation Accuracy: 0.86
- Epoch: 5, Loss: 1.82, Accuracy: 0.95
- Validation Loss: 1.81, Validation Accuracy: 0.93
- Epoch: 6, Loss: 1.54, Accuracy: 0.98
- Validation Loss: 1.62, Validation Accuracy: 0.93
- Epoch: 7, Loss: 1.45, Accuracy: 0.96
- Validation Loss: 1.44, Validation Accuracy: 0.93
- Epoch: 8, Loss: 1.37, Accuracy: 0.98
- Validation Loss: 1.28, Validation Accuracy: 0.93
- Epoch: 9, Loss: 1.30, Accuracy: 0.95
- Validation Loss: 1.19, Validation Accuracy: 1.00
- Epoch: 10, Loss: 1.34, Accuracy: 0.96
- Validation Loss: 1.16, Validation Accuracy: 1.00
- Finished Training

<img src="pics/11.png" style="height: 500px; width:1200px;"/>

# Build SVM
```
## SVM No Normalize and no MFCC
class SoundDS(Dataset):
    def __init__(self, audio_paths, labels, fixed_length=16000):
        self.audio_paths = audio_paths
        self.labels = labels
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[1] < self.fixed_length:
            pad_amount = self.fixed_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            waveform = waveform[:, :self.fixed_length]
        features = waveform.flatten().numpy()
        label = self.labels[idx]
        
        return features, label

file_list1 = os.listdir(path1)
file_list2 = os.listdir(path2)
c0_list = [os.path.join(path1, f) for f in file_list1 if os.path.splitext(f)[1] == '.wav']
c1_list = [os.path.join(path2, f) for f in file_list2 if os.path.splitext(f)[1] == '.wav']

audio_files = c0_list + c1_list
labels = [0] * len(c0_list) + [1] * len(c1_list)
myds = SoundDS(audio_files, labels)

num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

def prepare_data(dataloader):
    X = []
    y = []
    for features, label in dataloader:
        X.extend(features.numpy())  # Convert tensor to numpy array and extend
        y.extend(label.numpy())     # Convert tensor to numpy array and extend
    return np.array(X), np.array(y)

X_train, y_train = prepare_data(train_dl)
X_val, y_val = prepare_data(val_dl)

# # Normalize features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

train_accuracy = svm_model.score(X_train, y_train)
val_accuracy = svm_model.score(X_val, y_val)

print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')
```
- Training Accuracy: 1.0000
- Validation Accuracy: 0.8571

# Build KNN
```
def prepare_data(dataloader):
    X = []
    y = []
    for features, label in dataloader:
        X.extend(features.numpy())  # Convert tensor to numpy array and extend
        y.extend(label.numpy())     # Convert tensor to numpy array and extend
    return np.array(X), np.array(y)

X_train, y_train = prepare_data(train_dl)
X_val, y_val = prepare_data(val_dl)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

k_values = range(1, 5)
train_accuracies = []
val_accuracies = []

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    
    train_predictions = knn_model.predict(X_train)
    val_predictions = knn_model.predict(X_val)
    
    train_accuracy = accuracy_score(y_train, train_predictions)
    val_accuracy = accuracy_score(y_val, val_predictions)
    
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

plt.figure(figsize=(12, 6))
bar_width = 0.4
index = np.arange(len(k_values))

plt.bar(index, train_accuracies, bar_width, label='Training Accuracy', alpha=0.7)
plt.bar(index + bar_width, val_accuracies, bar_width, label='Validation Accuracy', alpha=0.7)

plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN Classifier Accuracy for Different Values of k')
plt.xticks(index + bar_width / 2, k_values)
plt.legend()
plt.grid(True)
plt.show()

# Best k
best_k = k_values[np.argmax(val_accuracies)]
print(f'Best k: {best_k}')
print(f'Best Validation Accuracy: {max(val_accuracies):.4f}')
```
<img src="pics/12.png" style="height: 495px; width:892px;"/>

- Best k: 1
- Best Validation Accuracy: 0.9286

# Build Random Forest
```
def prepare_data(dataloader):
    X = []
    y = []
    for features, label in dataloader:
        X.extend(features.numpy())  # Convert tensor to numpy array and extend
        y.extend(label.numpy())     # Convert tensor to numpy array and extend
    return np.array(X), np.array(y)

X_train, y_train = prepare_data(train_dl)
X_val, y_val = prepare_data(val_dl)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

train_predictions = rf_model.predict(X_train)
val_predictions = rf_model.predict(X_val)

train_accuracy = accuracy_score(y_train, train_predictions)
val_accuracy = accuracy_score(y_val, val_predictions)

print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')
print('\nClassification Report (Validation):\n')
print(classification_report(y_val, val_predictions))

importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:]  # Top 10 features

plt.figure(figsize=(10, 6))
plt.title('Top 10 Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), indices)
plt.xlabel('Relative Importance')
plt.show()
```

<img src="pics/13.png" style="height: 495px; width:892px;"/>



