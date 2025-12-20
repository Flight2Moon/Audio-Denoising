import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torcheval.metrics import PeakSignalNoiseRatio
from model_test import model_test

class InferenceModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        recon, _ = self.model(x)
        return recon

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

def log_weight_histograms(writer, model, model_name, epoch):
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f"{model_name}/{name}", param.data.cpu().numpy(), epoch)

def train_model(
    model,
    training_dataloader,
    training_validation_dataloader,
    criterion_mse=None,
    criterion_l1=None,
    criterion_huber=None,
    criterion_custom=None,
    learning_rate=None,
    scheduler=None,
    n_epochs=10,
    device="cuda",
    batch_size=1,
    test_dataloader=None,
    #input_size=128,
    input_size=64,
    writer=SummaryWriter(),
):
    model.to(device)
    dummy_data = torch.rand((1, 1, input_size[0], input_size[1])).to(device)
    writer.add_graph(InferenceModelWrapper(model), dummy_data)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    t_0 = max(10, n_epochs // 3)
    t_up = max(2, int(0.1 * t_0))
    eta_max = min(1e-3, learning_rate * 10)
    scheduler_inst = scheduler(
        optimizer=optimizer,
        T_0=t_0,
        T_mult=2,
        eta_max=eta_max,
        T_up=t_up,
        gamma=0.8,
        last_epoch=-1,
    )

    print("\n\n --- Training # 1 --- \n\n")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(training_dataloader)
        psnr_metric = PeakSignalNoiseRatio()

        for batch in progress:
            input_data = batch[0].to(device)
            target_data = batch[1].to(device)

            optimizer.zero_grad()

            output, loss = model(
                input_data,
                label_data=target_data,
                critation_1=criterion_mse,
                critation_2=criterion_custom,
                critation_3=criterion_l1,
            )

            loss.backward()
            optimizer.step()
            scheduler_inst.step()

            epoch_loss += loss.item()
            for i in range(len(input_data)):
                psnr_metric.update(output[i], target_data[i])

            progress.set_description(
                f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}"
            )

        for i in range(min(3, input_data.size(0))):
            log_image(writer, f"[Training]Input_{i}", input_data[i], global_step=epoch)
            log_image(writer, f"[Training]Reconstructed_{i}", output[i], global_step=epoch)

        log_weight_histograms(writer, model, model_name="[Training] UNet", epoch=epoch)
        psnr_result = psnr_metric.compute()
        writer.add_scalar("Loss/Train_Total", epoch_loss / len(training_dataloader), epoch + 1)
        writer.add_scalar("PSNR/Train", psnr_result, epoch + 1)
        #writer.add_scalar("LR", scheduler_inst.get_last_lr()[0], epoch + 1)

        if epoch >= n_epochs - 5:
            flattened_out = output.view(output.size(0), -1)
            flattened_in = input_data.view(input_data.size(0), -1)
            combined = torch.cat([flattened_in, flattened_out])
            metadata = ['input'] * output.size(0) + ['reconstructed'] * input_data.size(0)
            writer.add_embedding(
                combined,
                metadata=metadata,
                global_step=epoch,
                tag='[Training] Input vs Reconstructed'
            )

        print(
            f"\n\n------- TRAINING - I/O PAIR RESULT -------\n"
            f"Epoch [{epoch + 1}/{n_epochs}] Loss: {epoch_loss / len(training_dataloader):.4f}, PSNR: {psnr_result.item():.4f}\n"
        )

    ##########################################################################################

        print("\n\n --- Training # 2 --- \n\n")

        epoch_loss = 0.0
        progress = tqdm(training_validation_dataloader)
        psnr_metric = PeakSignalNoiseRatio()

        for batch in progress:
            input_data = batch[0].to(device)
            target_data = batch[1].to(device)

            optimizer.zero_grad()

            output, loss = model(
                input_data,
                label_data=target_data,
                critation_1=criterion_mse,
                critation_2=criterion_custom,
                critation_3=criterion_l1,
            )

            loss.backward()
            optimizer.step()
            scheduler_inst.step()

            epoch_loss += loss.item()
            for i in range(len(input_data)):
                psnr_metric.update(output[i], target_data[i])

            progress.set_description(
                f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}"
            )

        for i in range(min(3, input_data.size(0))):
            log_image(writer, f"[Training2]Input_{i}", input_data[i], global_step=epoch)
            log_image(writer, f"[Training2]Reconstructed_{i}", output[i], global_step=epoch)

        log_weight_histograms(writer, model, model_name="[Training2] UNet", epoch=epoch)
        psnr_result = psnr_metric.compute()
        writer.add_scalar("Loss/Train_Total2", epoch_loss / len(training_dataloader), epoch + 1)
        writer.add_scalar("PSNR/Train2", psnr_result, epoch + 1)
        #writer.add_scalar("LR", scheduler_inst.get_last_lr()[0], epoch + 1)

        if epoch >= n_epochs - 5:
            flattened_out = output.view(output.size(0), -1)
            flattened_in = input_data.view(input_data.size(0), -1)
            combined = torch.cat([flattened_in, flattened_out])
            metadata = ['input'] * output.size(0) + ['reconstructed'] * input_data.size(0)
            writer.add_embedding(
                combined,
                metadata=metadata,
                global_step=epoch,
                tag='[Training] Input vs Reconstructed2'
            )

        print(
            f"\n\n------- TRAINING - I/O SAME RESULT -------\n"
            f"Epoch [{epoch + 1}/{n_epochs}] Loss: {epoch_loss / len(training_dataloader):.4f}, PSNR: {psnr_result.item():.4f}\n"
        )

        model.eval()
        model_test(
            model=model,
            test_dataloader=test_dataloader,
            criterion_mse=criterion_mse,
            criterion_l1=criterion_l1,
            criterion_huber=criterion_huber,
            criterion_custom=criterion_custom,
            device=device,
            writer=writer,
            epoch=epoch + 1,
            max_epoch=n_epochs,
        )

    writer.close()
