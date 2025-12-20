import os
import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
from diffwave.src.diffwave.model import DiffWave
from types import SimpleNamespace

# ==== 모델 파라미터 및 환경 ====
params = SimpleNamespace(
    residual_layers=30,
    residual_channels=64,
    dilation_cycle_length=10,
    noise_schedule=torch.linspace(1e-4, 0.05, 50),
    conditional=False,
    unconditional=True,
    audio_in_channels=1,
    audio_out_channels=1,
    residual_stack_kernel_size=3,
    n_mels=128,
)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ==== 모델 로딩 ====
model = DiffWave(params).to(device)
model.load_state_dict(torch.load("./trained_models/slider_exp1/diffwave_finetune_epoch20.pth", map_location=device))
model.eval()


@torch.no_grad()
def inference_diffwave(model, noisy_audio, num_steps=50):
    x = noisy_audio.to(device)
    print("[DEBUG] inference_diffwave input x.shape:", x.shape)  # [1, 64000]
    for t in reversed(range(num_steps)):
        diffusion_step = torch.tensor([t], device=device)
        noise_pred = model(x, diffusion_step)
        x = x - noise_pred
    return x.squeeze().cpu().numpy()

def plot_waveform_and_spectrogram(noisy, denoised, sr, save_path):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    axs[0, 0].plot(noisy)
    axs[0, 0].set_title("Noisy Audio - Waveform")
    axs[0, 1].plot(denoised)
    axs[0, 1].set_title("Denoised Audio - Waveform")

    # [수정] 아래 두 줄의 인자 방식
    noisy_mel = librosa.feature.melspectrogram(y=noisy, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    noisy_logmel = librosa.power_to_db(noisy_mel, ref=np.max)
    axs[1, 0].imshow(noisy_logmel, origin="lower", aspect="auto", cmap=None)
    axs[1, 0].set_title("Noisy - Log-Mel Spectrogram")

    denoised_mel = librosa.feature.melspectrogram(y=denoised, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    denoised_logmel = librosa.power_to_db(denoised_mel, ref=np.max)
    axs[1, 1].imshow(denoised_logmel, origin="lower", aspect="auto", cmap=None)
    axs[1, 1].set_title("Denoised - Log-Mel Spectrogram")
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Visualization saved to: {save_path}")

def main():
    sr = 16000
    input_wav = "/home/jihoney/workdir/main_workdir/audio_denoising/dataset/unziped/development_dataset/slider/test/section_00_source_test_anomaly_0032_noAttribute.wav"
    output_wav = "./inference_result/example_noisy.wav"
    output_fig = "./inference_result/example_visualization.png"

    y, _ = librosa.load(input_wav, sr=sr)
    y = y[:sr * 4]
    y_tensor = torch.from_numpy(y).unsqueeze(0)  # [1, N] ONLY
    print("[DEBUG] main에서 y_tensor.shape:", y_tensor.shape)  # [1, 64000]

    denoised = inference_diffwave(model, y_tensor)
    denoised = denoised / (np.max(np.abs(denoised)) + 1e-6)

    torchaudio.save(output_wav, torch.from_numpy(denoised).unsqueeze(0), sample_rate=sr)
    print(f"[✓] Denoised audio saved: {output_wav}")

    plot_waveform_and_spectrogram(y, denoised, sr, output_fig)

if __name__ == "__main__":
    main()