"""
Sajat konvolucios neuralis halo (Custom CNN) COCO allat-osztalyozashoz.

Architektura attekintes:
    Bemenet: (B, 3, 224, 224)

    Jellemzokinyero resz – 5 konvolucios blokk:
        Block 1:  3  ->  32 csatorna  | 224 -> 112
        Block 2:  32 ->  64 csatorna  | 112 ->  56
        Block 3:  64 -> 128 csatorna  |  56 ->  28
        Block 4: 128 -> 256 csatorna  |  28 ->  14
        Block 5: 256 -> 512 csatorna  |  14 ->   7

    Minden blokk: Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> ReLU -> MaxPool(2x2)

    Osztalyozo resz:
        Global Average Pooling (7x7 -> 1x1)
        Linear(512, 512) -> ReLU -> Dropout(0.5)
        Linear(512, num_classes)

Parameterszam: ~7.2 million (num_classes=10 eseten)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Dupla konvolucios blokk:
        Conv(3x3, padding=1) -> BN -> ReLU -> Conv(3x3, padding=1) -> BN -> ReLU
        -> MaxPool(2x2) [-> Dropout2d]

    A MaxPool ket felezi a featuremap meretet mindket iranyban.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Sajat konvolucios neuralis halo allat-kepek osztalyozasahoz.

    Architektura:
        5 konvolucios blokk (novekvo csatornaszammal es Dropout regularizacioval)
        Global Average Pooling
        Ket teljesen csatolt reteg Dropout-tal

    Args:
        num_classes: Kimeneti osztályok szama (COCO allatokon: 10).
        dropout_fc:  Dropout valoszinuseg az FC retegeken.
    """

    def __init__(self, num_classes: int = 10, dropout_fc: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   32,  dropout_rate=0.10),   # 224x224 -> 112x112
            ConvBlock(32,  64,  dropout_rate=0.10),   # 112x112 ->  56x56
            ConvBlock(64,  128, dropout_rate=0.20),   #  56x56  ->  28x28
            ConvBlock(128, 256, dropout_rate=0.20),   #  28x28  ->  14x14
            ConvBlock(256, 512, dropout_rate=0.30),   #  14x14  ->   7x7
        )

        # Global Average Pooling: 7x7 -> 1x1 (minden csatornara)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming He inicializalas konvolucios retegekre."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)   # (B, 512, 7, 7)
        x = self.gap(x)        # (B, 512, 1, 1)
        x = self.classifier(x) # (B, num_classes)
        return x

    def count_parameters(self) -> int:
        """Tanithato parameterek szama."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
