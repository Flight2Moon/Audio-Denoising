import torch
import torch.nn as nn

class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity() 


        self.relu = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + identity
        return self.relu(out)

class UNetEncoder(nn.Module):
    def __init__(self):
        super(UNetEncoder, self).__init__()
        self.enc1 = UNetResidualBlock(1, 64)
        self.enc2 = UNetResidualBlock(64, 128)
        self.enc3 = UNetResidualBlock(128, 256)
        self.enc4 = UNetResidualBlock(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return [e1, e2, e3, e4]

class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = UNetResidualBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = UNetResidualBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNetResidualBlock(128, 64)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, enc_outputs):
        e1, e2, e3, e4 = enc_outputs
        d4 = self.up4(e4)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        return torch.sigmoid(self.out_conv(d2))




class UNetAutoEncoder(nn.Module):
    def __init__(self):
        super(UNetAutoEncoder, self).__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.semi_denoised_encoder = UNetEncoder()
        self.semi_denoised_decoder = UNetDecoder()

    def forward(
        self, 
        x,
        label_data=None,
        critation_1=None, 
        critation_2=None, 
        critation_3=None,
        loss_weights=[0.4, 0.4, 0.2],
    ):
        # enc_outputs = self.encoder(x)
        # recon = self.decoder(enc_outputs)

        e1, e2, e3, e4 = self.encoder(x)
        decoder_input = e1, e2, e3, e4
        semi_denoised_output = self.decoder(decoder_input)

        e1_semi, e2_semi, e3_semi, e4_semi = self.semi_denoised_encoder(semi_denoised_output)
        # semi_denoised_decoder_input = e1_semi, e2_semi, e3, e4
        semi_denoised_decoder_input = e1_semi, e2_semi, e3_semi, e4_semi # structure_fixedv2_fully
        denoised_output = self.semi_denoised_decoder(semi_denoised_decoder_input)

        # legacy model's output = recon

        # Inference Mode
        if label_data is None or critation_1 is None:
            return denoised_output, None

        loss = critation_1(label_data, denoised_output)

        if critation_2 is not None:
            # for the ssim l
            loss = loss * loss_weights[0] + critation_2(label_data, denoised_output) * loss_weights[1]

        if critation_3 is not None:
            loss = loss +  critation_3(label_data, denoised_output) * loss_weights[2]

        return denoised_output, loss


# class UNetEncoder_2(nn.Module):
#     def __init__(self):
#         super(UNetEncoder_2, self).__init__()
#         self.enc1 = UNetResidualBlock(1, 256)
#         self.enc2 = UNetResidualBlock(256, 512)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, x):
#         e1 = self.pool(self.enc1(self.pool(x)))
#         e2 = self.enc2(self.pool(e1))
        
#         return [e1, e2]
