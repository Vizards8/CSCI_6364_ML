import torch
import torch.nn as nn


class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(num_features=16),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(num_features=32),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(num_features=64),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


if __name__ == '__main__':
    model = CNN3D().cuda()
    input_tensor = torch.randn(32, 1, 48, 64, 64).cuda()

    from torchsummary import summary

    print(summary(model, (1, 48, 64, 64)))
    out = model(input_tensor)
    print(out.shape)
