from Neural_Network.network import Spec2Seq
from Neural_Network.train_function import train, test
from Utils.utils_dataset import create_dataloaders, SoundDataset
from Utils.viewing import greedy_decoder, CERMetric, WERMetric
from torch.nn import CTCLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torchsummaryX


# See if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Working on {device}")

# Initialize model
model = Spec2Seq().to(device)
torchsummaryX.summary(model, torch.zeros((10, 1, 201, 100), dtype=torch.float32, device=device))


# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 4
LR = 1e-3
loss_fn = CTCLoss(blank=45).to(device)
optimizer = Adam(model.parameters(), LR)
scheduler = ReduceLROnPlateau(optimizer, 'min')

metrics = [
    CERMetric(),
    WERMetric()
]
decoder = greedy_decoder

# create train and validation datasets
ds = SoundDataset("Datasets/LibriSpeech/dev-clean-processed")
train_dl, test_dl = create_dataloaders(ds, batch_size=BATCH_SIZE, split=0.8)


# Train model
for epoch in range(1, EPOCHS+1):
    print(f"Epoch {epoch}")
    print("-"*20)
    train(train_dl, model, loss_fn, optimizer, metrics=metrics, decoder=decoder)
    test(train_dl, model, loss_fn, metrics, decoder, scheduler=scheduler)
    torch.save(model.state_dict(), f"Models/{epoch:0>4}-state_dict.pt")
