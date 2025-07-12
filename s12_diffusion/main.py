import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffwave.src.diffwave.model import DiffWave  # DiffWave 공식 레포의 model.py 사용
from utils.dataloader import DenoisingAudioDataset
from tqdm import tqdm
from types import SimpleNamespace


machine_class = ['bearing', 'fan', 'gearbox', 'slider',]
machine_class_number = 3
experiment_name = f'{machine_class[machine_class_number]}_exp2'


dataset = DenoisingAudioDataset(
        input_dir= f'/home/jihoney/workdir/main_workdir/audio_denoising/dataset/unziped/development_dataset/{machine_class[machine_class_number]}/train', 
        target_dir=f'/home/jihoney/workdir/main_workdir/audio_denoising/dataset/dcase_denoised_dataset/{machine_class[machine_class_number]}/train_wav',
        sr=16000, 
        max_length=64000
    )
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

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
model = DiffWave(params).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 65
criterion = nn.MSELoss()
T = 50  # noise_schedule 길이 (params.noise_schedule.size(0))

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for x, y in tqdm(dataloader):
        x = x.cuda()
        y = y.cuda()
        batch_size = x.size(0)
        diffusion_steps = torch.randint(0, T, (batch_size,), device=x.device)  # [batch]개 랜덤 timestep
        output = model(x, diffusion_steps)       # output: [B, 1, N]
        output = output.squeeze(1)               # [B, N]로 맞춤
        loss = criterion(output, y)              # target도 [B, N]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")
    torch.save(model.state_dict(), f'./trained_models/{experiment_name}/diffwave_finetune_epoch{epoch+1}.pth')