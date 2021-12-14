from typing import Generator
from Neural_Network.network import AudioClassifier, Spec2Seq
from Neural_Network.train_function import train, test
from Utils.utils_dataset import create_dataloaders, SoundDataset
from Utils.viewing import greedy_decoder, BertScoreMetric, CERMetric, WERMetric
from torch.nn import CTCLoss
from torch.optim import AdamW
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Working on {device}")

# model = AudioClassifier().to(device)
model = Spec2Seq().to(device)
print([a.requires_grad for a in model.parameters()])


EPOCHS = 10
BATCH_SIZE = 10
loss_fn = CTCLoss(blank=28).to(device)
optimizer = AdamW(model.parameters(), 1e-3)

metrics = [
    CERMetric()
]
decoder = greedy_decoder

ds = SoundDataset("Datasets/LibriSpeech/dev-clean-processed")
train_dl, test_dl = create_dataloaders(ds, batch_size=BATCH_SIZE, split=0.8)


for epoch in range(1, EPOCHS+1):
    print(f"Epoch {epoch}")
    print("-"*20)
    train(train_dl, model, loss_fn, optimizer, metrics=metrics, decoder=decoder)
    test(test_dl, model, loss_fn, metrics, decoder)


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
