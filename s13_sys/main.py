# Calculate Number Of Audios In One Audio File
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import warnings

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# tensorboard --logdir=./logs --port=6006

# --------------------------------------------------
from architecture.unet import UNetAutoEncoder  


#from utils.data_process import CustomDataset
from utils.data_process_log_mel import CustomDataset, IdentityDataset, CustomTestDataset
from utils.custom_loss import ssim
from utils.custom_lr import CosineAnnealingWarmUpRestarts
from training import train_model
from glob import glob
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

gpu_number = 1

def get_device():

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon (MPS)"
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_number}")
        # cuda:0 --> screen gpu_0
        # cuda:1 --> screen gpu_1
        # cuda --> screen training


        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        print('Error Code 0 : No GPU, No Training Go Home')
        sys.exit()
    
    print(f"Device Name : {device_name}")

    return device

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_list_shape(data):
    """
    Recursively determines the shape of a nested list that contains tensors.
    Assumes all tensors in the list have the same shape.
    
    :param data: Nested list containing tensors.
    :return: Tuple representing the shape.
    """
    if isinstance(data, list):
        if len(data) == 0:
            return (0,)  # Handle empty list case
        return (len(data),) + get_list_shape(data[0])  # Recurse into first element
    elif isinstance(data, torch.Tensor):
        return data.size()
    else:
        return ()  # Return empty tuple for unsupported types


def main():


    #### Model Configure ####
    batch_size = 8
    lr = 1e-3
    epochs = 100
    input_size = [256, 256]
    max_pairs_per_audio = 13
    experiments_name = 'dcase2023_dataset'


    seed_everything(3737)

    ############## Path Settings ##############
    #machine_class = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
    machine_class = ['bearing', 'fan', 'gearbox', 'slider',]
    index_start_point = 0
    index_end_point = 1 #len(machine_class)

    ################################################
    # For make pretrained model by autumatically
    ################################################
    for index in range(index_start_point, len(machine_class)):

        print(f'\n\n ========= Training #{index} : {machine_class[index]} ========= ')


        selected_machine = machine_class[index]
        print(f"\n\nTraining Machine Type : {selected_machine}")

    # /Users/jihoney/Documents/workdir/jihoney/research_exp/audio_denoising/dataset/unziped/development_dataset/bearing/train
    # /home/jihoney/workdir/main_workdir/audio_denoising/dataset/unziped/development_dataset
    # /home/jihoney/workdir/main_workdir/audio_denoising/dataset/dcase2023_task2_dataset/unzip/dev_data
        dataset_root_path = f'/home/jihoney/workdir/main_workdir/audio_denoising/dataset/dcase2023_task2_dataset/unzip/dev_data/{selected_machine}'
        dataset_train_path = dataset_root_path + "/train" + "/*.wav"
        dataset_test_path = dataset_root_path  + "/test" + "/*.wav"

        all_files = sorted(glob(dataset_train_path))
        pair_files, identity_files = train_test_split(all_files, test_size=0.3, random_state=42)


        model_save_path = "./trained_models"  # Add a directory for saving models
        os.makedirs(model_save_path, exist_ok=True)  # Ensure the directory exists


        print('\n ========= information ========= ')
        device = get_device()

        #writer = SummaryWriter(log_dir=f'exp_test_{batch_size}b_{epochs}e_{device}_{selected_machine}_pairs{3}_256size')
        writer = SummaryWriter(log_dir=f'./experiments/{experiments_name}_{batch_size}b_{epochs}e_{device}_{selected_machine}_pairs{max_pairs_per_audio}_{input_size[0]}x{input_size[1]}size')

        #dataset_load_path = "/home/jihoney/huge_workdir/audio_event_detection/dataset/Honey_Dataset/overlayed_toyadmos_1/*.wav"
        print(dataset_train_path)
        print(dataset_test_path)

        dataset = CustomDataset(
            datas = pair_files, 
            resize_shape = (input_size[0], input_size[1]),
            audio_length = 10,
            writer = writer,
            max_pairs_per_audio = max_pairs_per_audio,
        )
            #sample_rate = config.stft_sample_rate, 

        #### ================================
        validation_dataset = IdentityDataset(
            file_list = identity_files,
            resize_shape = (input_size[0], input_size[1]),
        )
        #### ================================

        training_dataloader = DataLoader(
            dataset, 
            batch_size = batch_size, 
            shuffle = True,
            num_workers=2,  
            persistent_workers=True,
        )

        training_validation_dataloader = DataLoader(
            validation_dataset,
            batch_size = batch_size,
            shuffle = True,
        )

        ############### Test Dataloader ###############

        test_dataset = CustomTestDataset(
            dataset_root_path = dataset_test_path, 
            resize_shape = (input_size[0], input_size[1]),
            audio_length = 10,
            writer = writer,
            max_pairs_per_audio = 1,
        )

        test_dataloader = DataLoader(
            test_dataset, 
            batch_size = batch_size, 
            shuffle = True,
            num_workers=2,  
            persistent_workers=True,
        )

        # [0] For Using Single GPU
        if torch.cuda.device_count() == 1 or "RANK" not in os.environ:
            model = UNetAutoEncoder()
            model = model.to(f"cuda:{gpu_number}")
    
        # [1] For Using Dual GPU to training
        else:   
            #initializing        
            rank = int(os.environ["RANK"])
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")

            if not dist.is_initialized():
                dist.init_process_group("nccl")

            model = UNetAutoEncoder().to(rank)
            model = DDP(model, device_ids=[rank])
            # Running Command : torchrun --nproc_per_node=2 main.py


        train_model(
                model = model,
                training_dataloader = training_dataloader,
                training_validation_dataloader = training_validation_dataloader,
                criterion_mse = nn.MSELoss(),
                criterion_l1 = nn.L1Loss(),
                criterion_huber = nn.HuberLoss(),
                criterion_custom = ssim,
                learning_rate = lr,
                scheduler = CosineAnnealingWarmUpRestarts,
                n_epochs = epochs,
                device = device,
                batch_size = batch_size,
                test_dataloader = test_dataloader,
                input_size = input_size,
                writer = writer,
        )

        torch.save(model.state_dict(), os.path.join(model_save_path, f'{experiments_name}_{batch_size}b_{epochs}e_{selected_machine}_{max_pairs_per_audio}pairs_{input_size[0]}x{input_size[1]}.pt'))
        
if __name__ == "__main__":
    main()