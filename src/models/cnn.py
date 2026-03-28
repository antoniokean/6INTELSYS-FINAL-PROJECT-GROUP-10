from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=True):
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden_channels = max(16, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        scale = self.layers(self.pool(inputs))
        return inputs * scale


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = ConvNormAct(out_channels, out_channels, kernel_size=3, stride=1, activation=False)
        self.se = SqueezeExcite(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else ConvNormAct(in_channels, out_channels, kernel_size=1, stride=stride, activation=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs):
        residual = self.shortcut(inputs)
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.se(outputs)
        outputs = self.dropout(outputs)
        return self.activation(outputs + residual)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, 32, kernel_size=3, stride=1),
            ConvNormAct(32, 32, kernel_size=3, stride=1),
        )
        self.features = nn.Sequential(
            ResidualBlock(32, 32, stride=1, dropout=0.02),
            ResidualBlock(32, 64, stride=2, dropout=0.04),
            ResidualBlock(64, 96, stride=2, dropout=0.06),
            ResidualBlock(96, 128, stride=2, dropout=0.08),
            ResidualBlock(128, 160, stride=(2, 1), dropout=0.10),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(160),
            nn.Dropout(0.35),
            nn.Linear(160, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs):
        outputs = self.stem(inputs)
        outputs = self.features(outputs)
        outputs = self.pool(outputs)
        return self.classifier(outputs)
