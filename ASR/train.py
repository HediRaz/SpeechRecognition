from Neural_Network.network import AudioClassifier, Spec2Seq
from Neural_Network.train_function import train, test
from Utils.utils_dataset import create_dataloaders, SoundDataset
from torch.nn import CTCLoss
from torch.optim import AdamW
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Working on {device}")

# model = AudioClassifier().to(device)
model = Spec2Seq().to(device)

EPOCHS = 10
BATCH_SIZE = 16
loss_fn = CTCLoss()
optimizer = AdamW(model.parameters(), 1e-3)

ds = SoundDataset("Datasets/LibriSpeech/dev-clean-processed")
train_dl, test_dl = create_dataloaders(ds, batch_size=BATCH_SIZE, split=0.8)


for epoch in range(1, EPOCHS+1):
    print(f"Epoch {epoch}")
    print("-"*20)
    train(train_dl, model, loss_fn, optimizer)


if __name__ == "__main__":
    import torch
    import numpy as np
    from Utils.utils_dataset import int_list_to_ipa


    audio = torch.load("Datasets/LibriSpeech/dev-clean-processed/84/121123/84-121123-0000-audio.pt")
    audio = torch.unsqueeze(audio, 0)
    model.eval()
    model = model.to("cpu")
    with torch.no_grad():
        pred = model(audio)
        pred = torch.softmax(pred, -1)
        pred = torch.argmax(pred, -1)
        pred = torch.squeeze(pred)
        pred = torch.unique_consecutive(pred, dim=-1)
        # pred = [i for i in pred if i != 0]
        pred = pred.numpy()
        print(pred)
    pred = int_list_to_ipa(pred)
    print(pred)

    label = np.load("Datasets/LibriSpeech/dev-clean-processed/84/121123/84-121123-0000-label.npy")
    print(label)
    print(int_list_to_ipa(label))
