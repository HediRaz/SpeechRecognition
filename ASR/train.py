from typing import Generator
from Neural_Network.network import AudioClassifier, Spec2Seq
from Neural_Network.train_function import train, test
from Utils.utils_dataset import create_dataloaders, SoundDataset
from Utils.viewing import greedy_decoder, BertScoreMetric, CERMetric, WERMetric
from torch.nn import CTCLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torchsummaryX


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Working on {device}")

# model = AudioClassifier().to(device)
model = Spec2Seq().to(device)
torchsummaryX.summary(model, torch.zeros((10, 1, 201, 100), dtype=torch.float32, device=device))


EPOCHS = 100
BATCH_SIZE = 16
loss_fn = CTCLoss(blank=45).to(device)
optimizer = AdamW(model.parameters(), 1e-3)
scheduler = ReduceLROnPlateau(optimizer, 'min')

metrics = [
    CERMetric(),
    WERMetric()
]
decoder = greedy_decoder

ds = SoundDataset("Datasets/LibriSpeech/dev-clean-processed")
train_dl, test_dl = create_dataloaders(ds, batch_size=BATCH_SIZE, split=0.8)


for epoch in range(1, EPOCHS+1):
    print(f"Epoch {epoch}")
    print("-"*20)
    train(train_dl, model, loss_fn, optimizer, metrics=metrics, decoder=decoder, scheduler=scheduler)
    test(train_dl, model, loss_fn, metrics, decoder, scheduler=scheduler)


if __name__ == "__main__":
    import torch
    import numpy as np
    from Utils.utils_dataset import int_list_to_ipa
    from Utils.utils_dataset import int_list_to_char


    audio = torch.load("Datasets/LibriSpeech/dev-clean-processed/84/121123/84-121123-0000-audio.pt").unsqueeze(0)
    audio = torch.unsqueeze(audio, 0)
    model.eval()
    model = model.to("cpu")
    with torch.no_grad():
        pred = model(audio)[0]
        print(pred.shape)
        pred = decoder(pred)
    print(pred)

    label = np.load("Datasets/LibriSpeech/dev-clean-processed/84/121123/84-121123-0000-label.npy")
    print(label.shape)
    print(label)
    print(int_list_to_char(label))
