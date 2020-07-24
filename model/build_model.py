import torch
import torch.nn as nn

class conv3x3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self):
        super(down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        x = self.down(x)
        return x

class conv1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Simple_CNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Simple_CNN, self).__init__()

        self.conv_1 = conv3x3(n_channels, 12)
        self.pool_1 = down()
        self.conv_2 = conv3x3(12, 24)
        self.pool_2 = down()
        self.flaten = conv1x1(24, n_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.flaten(x)
        x = torch.squeeze(x)
        return torch.sigmoid(x)