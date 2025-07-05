import torch
import torch.nn as nn
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from architecture.recons_audio.unet import UNetAutoEncoder
from audio_denoising.workspace.jihoney_legacy.s08_sys.utils.legacy.data_process import audio2stft

def log_image(writer, tag, image_tensor, global_step, cmap='magma'):
    image = image_tensor.detach().cpu().numpy()
    image = np.squeeze(image)

    fig = plt.figure(figsize=(10, 4))
    librosa.display.specshow(image, sr=16000, hop_length=256, x_axis='time', y_axis='linear', cmap=cmap)
    plt.title(tag)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    writer.add_figure(tag, fig, global_step=global_step)
    plt.close(fig)

def audio2stft_safe(raw_audio):
    raw_audio = raw_audio / np.max(np.abs(raw_audio))
    stft_data = librosa.stft(y=raw_audio, n_fft=128, hop_length=256, window='hann')
    stft_data = np.abs(stft_data)
    stft_db = librosa.amplitude_to_db(stft_data, ref=np.max)
    stft_db = np.clip(stft_db, a_min=-80, a_max=0)
    stft_db = (stft_db + 80) / 80  # Normalize to [0, 1]
    return stft_db

def inference_tensorboard(model_path, audio_path, log_dir="./inference", step=0, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1. Load model
    model = UNetAutoEncoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load audio and convert to STFT
    raw_audio, sr = librosa.load(audio_path, sr=16000)
    stft_spec = audio2stft_safe(raw_audio[:sr * 10])  
    input_tensor = torch.FloatTensor(stft_spec).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    resize_transform = transforms.Resize((256, 256))
    input_tensor = resize_transform(input_tensor).to(device)

    # 3. TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # 4. Inference
    with torch.no_grad():
        output_tensor, _ = model(input_tensor)

    # 5. Log both input & output spectrogram
    log_image(writer, "Inference/Input_STFT", input_tensor[0], global_step=step)
    log_image(writer, "Inference/Output_STFT", output_tensor[0], global_step=step)

    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")

if __name__ == "__main__":

    pre_trained_model_name = 'trained_unet_model_16b_65e_slider.pt' # trained_unet_model_16b_65e_slider.pt
    selected_machine = 'slider' # 'bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve'
    data_class = 'test' # train, test
    audio_name = 'section_00_source_test_anomaly_0023_noAttribute'

    dataset_root_path = f'/Users/jihoney/Documents/workdir/jihoney/research_exp/audio_denoising/dataset/unziped/development_dataset/{selected_machine}/{data_class}'

    model_path =  f'/Users/jihoney/Documents/workdir/jihoney/research_exp/audio_denoising/workspace/jihoney_legacy/s07_sys/trained_models/{pre_trained_model_name}'
    audio_path = f"{dataset_root_path}/{audio_name}.wav"
    inference_tensorboard(model_path, audio_path)
