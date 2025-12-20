import torch
import torch.nn as nn
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torcheval.metrics import PeakSignalNoiseRatio

def log_image(writer, tag, image_tensor, global_step, cmap=None):
    image = image_tensor.detach().squeeze(0).cpu()
    height, width = image.shape[:2]
    dpi = 100
    figsize = (width / dpi, height / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, cmap=cmap)
    ax.axis('off')
    writer.add_figure(tag, fig, global_step=global_step)
    plt.close(fig)

def model_test(
    model,
    test_dataloader,
    criterion_mse,
    criterion_l1,
    criterion_huber,
    criterion_custom,
    device,
    writer,
    epoch,
    max_epoch,
):
    progress = tqdm(test_dataloader)
    psnr_input_audio = PeakSignalNoiseRatio()

    with torch.no_grad():
        total_loss = 0.0

        print('\n\n', '=' * 7, ' Test ', '=' * 7)
        for batch in progress:
            input_data = batch[0].to(device)
            target_data = batch[1].to(device)

            output, loss = model(
                input_data,
                target_data,
                criterion_mse,
                criterion_custom,
            )



            for i in range(len(input_data)):
                psnr_input_audio.update(output[i], input_data[i])

            total_loss += loss.item()

            progress.set_description(
                f"Test - Epoch {epoch} : Loss : {loss.item():.4f}"
            )

        for i in range(3):
            log_image(writer, f'[TEST]Input_Original - {i}', input_data[i], epoch)
            log_image(writer, f'[TEST]Input_Reconstructed - {i}', output[i], epoch)

        psnr_result = psnr_input_audio.compute()

        writer.add_scalar("Loss/Reconstruction_Test", total_loss / len(test_dataloader), epoch + 1)
        writer.add_scalar("PSNR_Input_Audio_Test", psnr_result, epoch + 1)

        if epoch >= max_epoch - 5:
            flat_output = output.view(output.size(0), -1)
            flat_input = input_data.view(input_data.size(0), -1)
            combined = torch.cat([flat_input, flat_output])
            metadata = ['input_data'] * input_data.size(0) + ['reconstructed'] * output.size(0)

            writer.add_embedding(
                combined,
                metadata=metadata,
                global_step=epoch,
                tag='[Test] Input vs Reconstructed'
            )

        print(
            f'\n\n------- TEST RESULT -------\n'
            f"Epoch [{epoch}] "
            f"Reconstruction Loss: {total_loss / len(test_dataloader)}\n"
            f"PSNR : {psnr_result.item():.4f}\n"
        )
        print('')