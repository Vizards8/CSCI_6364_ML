import torch
import torch.nn as nn


class VNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_filters=16):
        super().__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, num_filters, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_filters, num_filters, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = nn.Sequential(
            nn.Conv3d(num_filters, num_filters * 2, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_filters * 2, num_filters * 2, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters * 2),
            nn.ReLU(inplace=True)
        )

        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = nn.Sequential(
            nn.Conv3d(num_filters * 2, num_filters * 4, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_filters * 4, num_filters * 4, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters * 4),
            nn.ReLU(inplace=True)
        )

        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder4 = nn.Sequential(
            nn.Conv3d(num_filters * 4, num_filters * 8, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters * 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_filters * 8, num_filters * 8, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters * 8),
            nn.ReLU(inplace=True)
        )

        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(num_filters * 8, num_filters * 16, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters * 16),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_filters * 16, num_filters * 16, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters * 16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(num_filters * 16, num_filters * 8, kernel_size=2, stride=2),
            nn.BatchNorm3d(num_filters * 8),
            nn.ReLU(inplace=True)
        )

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose3d(num_filters * 16, num_filters * 8, kernel_size=2, stride=2),
            nn.BatchNorm3d(num_filters * 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_filters * 8, num_filters * 8, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters * 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_filters * 8, num_filters * 4, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters * 4),
            nn.ReLU(inplace=True)
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(num_filters * 8, num_filters * 4, kernel_size=2, stride=2),
            nn.BatchNorm3d(num_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_filters * 4, num_filters * 4, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_filters * 4, num_filters * 2, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters * 2),
            nn.ReLU(inplace=True)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(num_filters * 4, num_filters * 2, kernel_size=2, stride=2),
            nn.BatchNorm3d(num_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_filters * 2, num_filters * 2, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_filters * 2, num_filters, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(num_filters * 2, num_filters, kernel_size=2, stride=2),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_filters, num_filters, kernel_size=5, padding=2),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_filters, out_channels, kernel_size=1)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)
        enc2 = self.encoder2(pool1)
        pool2 = self.pool2(enc2)
        enc3 = self.encoder3(pool2)
        pool3 = self.pool3(enc3)
        enc4 = self.encoder4(pool3)
        pool4 = self.pool4(enc4)
        bottle = self.bottleneck(pool4)
        dec4 = self.decoder4(torch.cat([bottle, enc4], dim=1))
        dec3 = self.decoder3(torch.cat([dec4, enc3], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.decoder1(torch.cat([dec2, enc1], dim=1))
        return dec1


if __name__ == '__main__':
    model = VNet().cuda()
    input_tensor = torch.randn(32, 1, 48, 64, 64).cuda()

    from torchsummary import summary

    print(summary(model, (1, 48, 64, 64)))
    out = model(input_tensor)
    print(out.shape)
