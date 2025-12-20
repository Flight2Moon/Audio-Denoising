import os
import sys
import torch
import numpy as np
import librosa
import torch.nn.functional as F
from tqdm import tqdm

# Load STFT from raw waveform
# def audio_to_stft(raw_audio, n_fft=128, hop_length=256):
#     raw_audio = raw_audio / np.max(np.abs(raw_audio))
#     stft_data = librosa.stft(y=raw_audio, n_fft=n_fft, hop_length=hop_length, window="hann")
#     stft_data = np.abs(stft_data)
#     stft_db = librosa.amplitude_to_db(stft_data, ref=np.max)
#     stft_norm = (stft_db - np.min(stft_db)) / (np.max(stft_db) - np.min(stft_db))
#     return stft_norm

def audio2logmel(raw_audio, sr=16000, n_mels=256, n_fft=2048, hop_length=250):
    # Normalize
    raw_audio = raw_audio / np.max(np.abs(raw_audio))
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=raw_audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    
    # Convert to log scale (dB)
    logmel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    logmel = (logmel - np.min(logmel)) / (np.max(logmel) - np.min(logmel) + 1e-6)
    
    # Just in case: pad or crop to exactly (256, 640)
    logmel = logmel[:, :640] if logmel.shape[1] >= 640 else np.pad(logmel, ((0,0),(0,640-logmel.shape[1])), mode='constant')
    
    return logmel

# # Load audio and return STFT
# def path_to_feature(path, sr=16000, duration=10):
#     y, _ = librosa.load(path, sr=sr)
#     y = y[:sr * duration]
#     return audio_to_stft(y)


def path2feature(path): 
    sr = 16000
    y, _ = librosa.load(path, sr=sr, duration=10.0)
    logmel = audio2logmel(y, sr=sr)
    return logmel



# Import model
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from architecture.recons_audio.unet import UNetAutoEncoder


def load_model(model_path, device):
    model = UNetAutoEncoder()
    state_dict = torch.load(model_path, map_location=device)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def process_audio_to_npy(audio_path, model, input_size, device, output_dir):
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    log_mel = path2feature(audio_path)

    tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).float().to(device)
    input_tensor = F.interpolate(tensor, size=(input_size, input_size), mode="bilinear", align_corners=False)

    with torch.no_grad():
        output = model(input_tensor)
        denoised_output = output[0] if isinstance(output, tuple) else output

    denoised_np = denoised_output.squeeze().cpu().numpy()
    save_path = os.path.join(output_dir, f"{basename}.npy")
    np.save(save_path, denoised_np)


def main():
    # /home/jihoney/workdir/main_workdir/audio_denoising/workspace/s11_sys/trained_models/s08v_16b_100epoch/trained_denoiser_s08v_16b_100e_bearing_10pairs_256x256.pt
    dataset_audio_class = "slider"
    input_audio_dir = (
        f"/home/jihoney/workdir/main_workdir/audio_denoising/dataset/unziped/"
        f"development_dataset/{dataset_audio_class}/train"
    )
    output_npy_dir = f"../../../dataset/dcase_denoised_dataset/{dataset_audio_class}/train"
    model_ckpt = (
        #f"../trained_models/trained_denoiser_s08v_16b_65e_{dataset_audio_class}_10pairs_256x256.pt"
        #f"../trained_models/trained_denoiser_s08v_16b_65e_{dataset_audio_class}_10pairs_256x256.pt"
        f"../trained_models/s08v_16b_100epoch/trained_denoiser_s08v_16b_100e_{dataset_audio_class}_10pairs_256x256.pt"
    )
    input_size = 256
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_npy_dir, exist_ok=True)
    model = load_model(model_ckpt, device)

    wav_files = [f for f in os.listdir(input_audio_dir) if f.endswith(".wav")]
    for wav_file in tqdm(wav_files, desc="Generating denoised log-mel .npy"):
        audio_path = os.path.join(input_audio_dir, wav_file)
        process_audio_to_npy(audio_path, model, input_size, device, output_npy_dir)


if __name__ == "__main__":
    main()
