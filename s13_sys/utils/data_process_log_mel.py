import random
import os
import torchvision.transforms as transforms
import soundfile as sf
import numpy as np
import librosa 
import torch
import warnings

from torch.utils.tensorboard import SummaryWriter
from glob import glob
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# def audio2stft(raw_audio): 
#     raw_audio = raw_audio / np.max(raw_audio)
#     stft_data = librosa.stft(
#         y = raw_audio,
#         n_fft = 128,
#         hop_length = 256,
#         window='hann',
#     )
#     stft_data = abs(stft_data)
#     stft_data = librosa.amplitude_to_db(stft_data, ref=np.max)
#     stft_data = (stft_data - np.min(stft_data)) / (np.max(stft_data) - np.min(stft_data))
#     return stft_data

# def audio2logmel(raw_audio, sr=16000, n_mels=128, n_fft=1024, hop_length=512):
#     # Normalize
#     raw_audio = raw_audio / np.max(np.abs(raw_audio))
    
#     # Compute mel spectrogram
#     mel_spec = librosa.feature.melspectrogram(
#         y=raw_audio,
#         sr=sr,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         n_mels=n_mels,
#         power=2.0
#     )
    
#     # Convert to log scale (dB)
#     logmel = librosa.power_to_db(mel_spec, ref=np.max)
    
#     # Normalize to [0, 1]
#     logmel = (logmel - np.min(logmel)) / (np.max(logmel) - np.min(logmel) + 1e-6)

#     # Pad or crop to shape (n_mels, 640)
#     logmel = logmel[:, :640] if logmel.shape[1] >= 640 else np.pad(logmel, ((0,0),(0,640-logmel.shape[1])), mode='constant')
    
#     return logmel


def audio2logmel(raw_audio, sr=16000, n_mels=256, target_frames=256):
    raw_audio = raw_audio / (np.max(np.abs(raw_audio)) + 1e-6)

    hop_length = int(len(raw_audio) / target_frames)
    hop_length = max(1, hop_length)

    mel_spec = librosa.feature.melspectrogram(
        y=raw_audio, sr=sr,
        n_fft=2 * hop_length,
        hop_length=hop_length,
        n_mels=n_mels, power=2.0
    )
    logmel = librosa.power_to_db(mel_spec, ref=np.max)
    logmel = (logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-6)


    if logmel.shape[1] < target_frames:
        pad = target_frames - logmel.shape[1]
        logmel = np.pad(logmel, ((0,0),(0,pad)), mode='constant')
    else:
        logmel = logmel[:, :target_frames]

    assert logmel.shape == (n_mels, target_frames)
    return logmel

def path2feature(path):
    sr = 16000
    y, _ = librosa.load(path, sr=sr, duration=10.0)
    logmel = audio2logmel(raw_audio=y, sr=sr, n_mels=256, target_frames=256)
    return logmel 

# def path2feature(path): 
#     sr = 16000
#     y, _ = librosa.load(path, sr=sr, duration=10.0)
#     logmel = audio2logmel(y, sr=sr)
#     return logmel 


# def path2feature(path, sr=16000, duration=10.24):
#     target_len = int(sr * duration)
#     y, _ = librosa.load(path, sr=sr)

#     if len(y) < target_len:
#         raise ValueError("Audio too short for target duration.")

#     y = y[:target_len]
#     return audio2logmel(y, sr=sr)

    

class CustomDataset(Dataset): 
    def __init__(self, datas=None, resize_shape=(128, 128), audio_length=10, writer=SummaryWriter(), max_pairs_per_audio=4):
        self.resize_shape = resize_shape
        self.audio_length = audio_length
        self.writer = writer
        self.max_pairs_per_audio = max_pairs_per_audio

        selected_paths = datas
        
        self.dataset_root_path = selected_paths
        self.dataset_len = len(selected_paths)

        self.stft_cache = self._precompute_stfts()
        self.pairs = self._build_psnr_filtered_pairs()

        print(f'\n\n<<< Dataset Information >>>\n',
              f'Total Dataset Size : {len(self.pairs)}\n')

    def _precompute_stfts(self):
        stft_cache = {}
        for idx, path in enumerate(self.dataset_root_path):
            stft = path2feature(path)
            stft_cache[idx] = stft
        return stft_cache

    def _psnr(self, a, b, eps=1e-8):
        return 10 * np.log10(1.0 / (np.mean((a - b) ** 2) + eps))

    def _build_psnr_filtered_pairs(self):
        pairs = []
        for i in range(self.dataset_len):
            psnr_scores = []
            for j in range(self.dataset_len):
                if i == j:
                    continue
                score = self._psnr(self.stft_cache[i], self.stft_cache[j])
                psnr_scores.append((j, score))
            psnr_scores.sort(key=lambda x: x[1], reverse=True)
            candidate_indices = [j for j, _ in psnr_scores[:self.max_pairs_per_audio * 2]]
            sampled = random.sample(candidate_indices, k=min(self.max_pairs_per_audio, len(candidate_indices)))
            for j in sampled:
                pairs.append((i, j))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_audio_index, target_audio_index = self.pairs[index]
        input_audio_path = self.dataset_root_path[input_audio_index]
        target_audio_path = self.dataset_root_path[target_audio_index]

        resize_transform = transforms.Resize(self.resize_shape)

        input_audio = path2feature(input_audio_path)
        target_audio = path2feature(target_audio_path)

        input_tensor = resize_transform(torch.FloatTensor(input_audio).unsqueeze(0))
        target_tensor = resize_transform(torch.FloatTensor(target_audio).unsqueeze(0))

        return input_tensor, target_tensor
    
class IdentityDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, resize_shape, ):
        self.file_list = file_list
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        input_audio = path2feature(self.file_list[idx])
        resize_transform = transforms.Resize(self.resize_shape)
        input_tensor = resize_transform(torch.FloatTensor(input_audio).unsqueeze(0))

        return input_tensor, input_tensor
    


class CustomTestDataset(Dataset): 
    def __init__(self, dataset_root_path=None, resize_shape=(128, 128), audio_length=10, writer=SummaryWriter(), max_pairs_per_audio=4):
        self.resize_shape = resize_shape
        self.audio_length = audio_length
        self.writer = writer
        self.max_pairs_per_audio = max_pairs_per_audio

        selected_paths = list(glob(dataset_root_path))
        self.dataset_root_path = selected_paths
        self.dataset_len = len(selected_paths)

        self.stft_cache = self._precompute_stfts()
        self.pairs = self._build_psnr_filtered_pairs()

        print(f'\n\n<<< Dataset Information >>>\n',
              f'Total Dataset Size : {len(self.pairs)}\n')

    def _precompute_stfts(self):
        stft_cache = {}
        for idx, path in enumerate(self.dataset_root_path):
            stft = path2feature(path)
            stft_cache[idx] = stft
        return stft_cache

    def _psnr(self, a, b, eps=1e-8):
        return 10 * np.log10(1.0 / (np.mean((a - b) ** 2) + eps))

    def _build_psnr_filtered_pairs(self):
        pairs = []
        for i in range(self.dataset_len):
            psnr_scores = []
            for j in range(self.dataset_len):
                if i == j:
                    continue
                score = self._psnr(self.stft_cache[i], self.stft_cache[j])
                psnr_scores.append((j, score))
            psnr_scores.sort(key=lambda x: x[1], reverse=True)
            candidate_indices = [j for j, _ in psnr_scores[:self.max_pairs_per_audio * 2]]
            sampled = random.sample(candidate_indices, k=min(self.max_pairs_per_audio, len(candidate_indices)))
            for j in sampled:
                pairs.append((i, j))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_audio_index, target_audio_index = self.pairs[index]
        input_audio_path = self.dataset_root_path[input_audio_index]
        target_audio_path = self.dataset_root_path[target_audio_index]

        resize_transform = transforms.Resize(self.resize_shape)

        input_audio = path2feature(input_audio_path)
        target_audio = path2feature(target_audio_path)

        input_tensor = resize_transform(torch.FloatTensor(input_audio).unsqueeze(0))
        target_tensor = resize_transform(torch.FloatTensor(target_audio).unsqueeze(0))

        return input_tensor, target_tensor

