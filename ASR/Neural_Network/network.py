import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 1), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        # self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Conv1d(64*8, 45, 1)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = torch.unsqueeze(x, 1)
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        # x = self.ap(x)
        # N C H T
        x = x.view(x.size(0), x.size(1)*x.size(2), x.size(3))

        # Linear layer
        x = self.lin(x)

        # Final output
        return x

# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device


class MobileBlock(nn.Module):

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
