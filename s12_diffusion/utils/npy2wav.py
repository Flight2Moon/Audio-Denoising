import numpy as np
import librosa
import soundfile as sf
import glob
import os

machine_class = ['bearing', 'fan', 'gearbox', 'slider']
SR = 16000
N_FFT = 1024
HOP_LENGTH = 512      # 생성 시와 동일하게!
N_MELS = 128
FMAX = 8000           # 기본값과 명확히 일치하게!

LOGMEL_MIN = -80      # 생성 파라미터에 따라 다름 (기본 -80dB)
LOGMEL_MAX = 0        # power_to_db(ref=np.max)라면 0dB

for cls in machine_class:
    SEMI_DENOISED_DIR = f'/home/jihoney/workdir/main_workdir/audio_denoising/dataset/dcase_denoised_dataset/{cls}/train_1024_250'
    OUTPUT_WAV_DIR = f'/home/jihoney/workdir/main_workdir/audio_denoising/dataset/dcase_denoised_dataset/{cls}/train_wav'
    os.makedirs(OUTPUT_WAV_DIR, exist_ok=True)

    for npy_path in glob.glob(os.path.join(SEMI_DENOISED_DIR, '*.npy')):
        logmel_norm = np.load(npy_path)
        # 0~1 정규화 해제
        logmel = logmel_norm * (LOGMEL_MAX - LOGMEL_MIN) + LOGMEL_MIN
        # log-mel(dB) → mel(power)
        mel = librosa.db_to_power(logmel, ref=1.0)
        # mel → wav 변환, 모든 파라미터 일치!
        wav = librosa.feature.inverse.mel_to_audio(
            mel, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX
        )
        basename = os.path.splitext(os.path.basename(npy_path))[0]
        sf.write(os.path.join(OUTPUT_WAV_DIR, f'{basename}.wav'), wav, SR)