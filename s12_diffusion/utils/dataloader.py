from torch.utils.data import Dataset, DataLoader
import torch
import librosa
import glob
import numpy as np
import os

class DenoisingAudioDataset(Dataset):
    def __init__(self, input_dir, target_dir, sr=16000, max_length=64000):
        self.input_files = sorted(glob.glob(os.path.join(input_dir, '*.wav')))
        self.target_files = sorted(glob.glob(os.path.join(target_dir, '*.wav')))
        assert len(self.input_files) == len(self.target_files), "Input/target 길이 불일치"
        self.sr = sr
        self.max_length = max_length

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        x, _ = librosa.load(self.input_files[idx], sr=self.sr)
        y, _ = librosa.load(self.target_files[idx], sr=self.sr)
        if len(x) < self.max_length:
            x = np.pad(x, (0, self.max_length - len(x)))
        else:
            x = x[:self.max_length]
        if len(y) < self.max_length:
            y = np.pad(y, (0, self.max_length - len(y)))
        else:
            y = y[:self.max_length]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()