import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


class MobileBlock(nn.Module):
    """
    A block of MobileNet
    
    """

    def __init__(self, n):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(n),
            nn.GELU(),
            nn.Conv2d(n, 2*n, 1, 1),
            nn.BatchNorm2d(2*n),
            nn.GELU(),
            nn.Conv2d(2*n, 2*n, 3, 1, 1, 1, 2*n),
            nn.BatchNorm2d(2*n),
            nn.GELU(),
            nn.Conv2d(2*n, n, 1, 1),
        )    
    
    def forward(self, x):
        return self.model(x) + x


class Conformer(nn.Module):
    """
    A Conformer layer as introduced in https://arxiv.org/abs/2005.08100
    
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.feed_forward_1 = nn.Sequential(
            nn.LayerNorm([in_features]),
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Dropout(.1),
            nn.Linear(out_features, out_features),
            nn.Dropout(.1)
        )
        self.k = nn.Linear(out_features, out_features)
        self.q = nn.Linear(out_features, out_features)
        self.v = nn.Linear(out_features, out_features)
        self.MHA = nn.MultiheadAttention(out_features, num_heads=8, dropout=0, batch_first=True)
        self.lnorm = nn.LayerNorm(out_features)
        self.Convolutions = nn.Sequential(
            # B F T
            nn.Conv1d(out_features, 2*out_features, 1),
            nn.GELU(),
            nn.Conv1d(2*out_features, 2*out_features, 3, 1, 1, 1, 2*out_features, bias=False),
            nn.BatchNorm1d(2*out_features),
            nn.GELU(),
            nn.Conv1d(2*out_features, out_features, 1),
            nn.Dropout(.1)
        )
        self.feed_forward_2 = nn.Sequential(
            nn.LayerNorm([out_features]),
            nn.Linear(out_features, out_features),
            nn.GELU(),
            nn.Dropout(.1),
            nn.Linear(out_features, out_features),
            nn.Dropout(.1)
        )

    def forward(self, x):
        # B T F
        x = 0.5*self.feed_forward_1(x) + x
        x = self.MHA(self.k(x), self.q(x), self.v(x), need_weights=False)[0] + x
        x = self.lnorm(x)
        x = x.transpose(1, 2)
        # B F T
        x = self.Convolutions(x) + x
        x = x.transpose(1, 2)
        # B T F
        x = 0.5*self.feed_forward_2(x) + x
        return x


class Spec2Seq(nn.Module):
    """
    The complete speech recognition model

    CNN --> MLP --> Conformers --> MLP
    
    """

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # N x 1 x 201 x T
            nn.Conv2d(1, 8, 5, (1, 1)),
            # N x 16 x (T-4) x 198
            nn.GELU(),
            MobileBlock(8),
            # MobileBlock(8),
            # MobileBlock(8),
            nn.Conv2d(8, 8, 3, (2, 1)),
            # N x 16 x (T-6) x 98
            nn.GELU(),
            # MobileBlock(16),
            # MobileBlock(16),
            # MobileBlock(16),
            # nn.Conv2d(16, 32, 3, (2, 1)),
            # N x 32 x (T-8) x 48
            # nn.GELU(),
            # MobileBlock(32),
            # MobileBlock(32),
            # MobileBlock(32),
            # nn.Conv2d(32, 64, 3, (2, 1)),
            # N x 64 x (T-10) x 24
            # nn.ReLU(),
            # MobileBlock(64),
            # MobileBlock(64),
            # MobileBlock(64),
            # nn.Conv2d(64, 128, 3, (2, 1)),
            # N x 128 x (T-12) x 6
            # nn.ReLU(),
            # MobileBlock(128),
            # MobileBlock(128),
            # nn.MaxPool2d((6, 1), (6, 1))
        )

        self.mlp = nn.Sequential(
            nn.Linear(8*98, 128),
            nn.Dropout(.1),
            nn.GELU(),
            # nn.Linear(256, 32),
            # nn.GELU()
        )
        # self.lstm = nn.GRU(input_size=128, hidden_size=128, batch_first=True, bidirectional=True, num_layers=1)
        self.conformers = nn.Sequential(
            Conformer(128, 128),
            Conformer(128, 128),
            Conformer(128, 128)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(.1),
            nn.GELU(),
            nn.Linear(128, 46)
        )


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), x.size(1)*x.size(2), x.size(3))
        x = torch.transpose(x, 1, 2)
        x = self.mlp(x)
        # x, _ = self.lstm(x)
        x = self.conformers(x)
        x = self.classifier(x)
        return x
