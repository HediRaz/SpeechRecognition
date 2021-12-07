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
            nn.Conv2d(n, 2*n, 1, 1),
            nn.BatchNorm2d(2*n),
            nn.GELU(),
            nn.Conv2d(2*n, 2*n, 3, 1, 1, 1, 2*n),
            nn.BatchNorm2d(2*n),
            nn.GELU(),
            nn.Conv2d(2*n, n, 1, 1),
            nn.BatchNorm2d(n),
        )    
    
    def forward(self, x):
        return self.model(x) + x


class Spec2Seq(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # N x 1 x 128 x T
            nn.Conv2d(1, 8, 5, (1, 1)),
            # N x 8 x 124 x (T-4)
            nn.GELU(),
            MobileBlock(8),
            MobileBlock(8),
            MobileBlock(8),
            nn.Conv2d(8, 16, 3, (2, 1)),
            # N x 16 x 61 x (T-6)
            nn.GELU(),
            MobileBlock(16),
            MobileBlock(16),
            MobileBlock(16),
            nn.Conv2d(16, 32, 3, (2, 1)),
            # N x 32 x 29 x (T-8)
            nn.GELU(),
            MobileBlock(32),
            MobileBlock(32),
            MobileBlock(32),
            nn.Conv2d(32, 64, 3, (2, 1)),
            # N x 64 x 13 x (T-10)
            nn.GELU(),
            MobileBlock(64),
            MobileBlock(64),
            MobileBlock(64),
            nn.Conv2d(64, 128, 3, (2, 1)),
            # N x 128 x 5 x (T-12)
            nn.GELU(),
            MobileBlock(128),
            MobileBlock(128),
            nn.MaxPool2d((5, 1), (5, 1))
        )

        self.lstm = nn.GRU(input_size=128, output_size=128, batch_first=True, bidirectional=True, num_layers=4)

        self.classifier = nn.Sequential(
            nn.Linear(2*128, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )

    
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv_layers(x)
        x = torch.squeeze(x, 2)
        x = torch.transpose(x, 1, 2)
        x = self.lstm(x)
        x = self.classifier(x)
        return x
