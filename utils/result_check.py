import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import random
import glob

# ==== 설정 ====
sr = 16000
n_fft = 1024
hop_length = 512
n_mels = 64

# 기본 경로 설정
machine_class = 'slider'
section_name = 'section_00'
target_dir = '/home/jihoney/workdir/main_workdir/audio_denoising/dataset'
wav_dir = f"{target_dir}/unziped/development_dataset/{machine_class}/train"
npy_dir = f"{target_dir}/dcase_denoised_dataset/{machine_class}/train_640x128_final"
save_dir = f"./result_check/{machine_class}"
os.makedirs(save_dir, exist_ok=True)

# ==== 파일 리스트 생성 (file_list_generator 대체 간소화) ====
def get_matching_wav_npy_pairs(section_name):
    wav_query = os.path.join(wav_dir, f"{section_name}_*normal_*.wav")
    wav_files = sorted(glob.glob(wav_query))
    pairs = []

    for wav_path in wav_files:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        npy_path = os.path.join(npy_dir, f"{base}.npy")
        if os.path.exists(npy_path):
            pairs.append((wav_path, npy_path))
        else:
            print(f"⚠️ 대응되는 .npy 없음: {base}")
    return pairs

# ==== 시각화 함수 ====
def plot_comparison_save(wav_path, npy_path, save_path):
    # 1. denoised log-mel (.npy → 직접 사용)
    denoised_logmel = np.load(npy_path)
    print(f"[DEBUG] npy shape: {denoised_logmel.shape}")  # (Time, Mel)인지 확인

    # 2. baseline 입력용 log-mel (.wav → 계산)
    wav, _ = librosa.load(wav_path, sr=sr)
    baseline_mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
                                                  hop_length=hop_length, n_mels=n_mels)
    baseline_logmel = librosa.power_to_db(baseline_mel, ref=np.max)

    # 3. 시각화
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    librosa.display.specshow(baseline_logmel, sr=sr, hop_length=hop_length,
                              x_axis='time', y_axis='mel', ax=axs[0])
    axs[0].set_title("Baseline Input (from WAV)")

    librosa.display.specshow(denoised_logmel, sr=sr, hop_length=hop_length,
                              x_axis='time', y_axis='mel', ax=axs[1])
    axs[1].set_title("Denoised Result (from NPY)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==== 실행 함수 ====
def compare_N_examples(N):
    pairs = get_matching_wav_npy_pairs(section_name)
    random.shuffle(pairs)
    selected = pairs[:N]

    for wav_path, npy_path in selected:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        save_path = os.path.join(save_dir, f"{base}_compare.png")
        print(f"✅ 저장 중: {save_path}")
        plot_comparison_save(wav_path, npy_path, save_path)

# ==== 실행 ====
if __name__ == "__main__":
    compare_N_examples(N=3)