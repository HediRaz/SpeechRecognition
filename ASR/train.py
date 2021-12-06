from Neural_Network.network import AudioClassifier
from Neural_Network.train_function import train, test
from Utils.utils_dataset import create_dataloaders, SoundDataset
from torch.nn import CTCLoss
from torch.optim import Adam


model = AudioClassifier()

loss_fn = CTCLoss()
optimizer = Adam(model.parameters(), 1e-3)

ds = SoundDataset("Datasets/LibriSpeech/dev-clean-processed")
train_dl, test_dl = create_dataloaders(ds, batch_size=16, split=0.8)

train(train_dl, model, loss_fn, optimizer)


if __name__ == "__main__":
    import torch
    from Utils.utils_dataset import int_list_to_ipa
    audio = torch.load("Datasets/LibriSpeech/dev-clean-processed/84/121123/84-121123-0000-audio.pt")
    audio = torch.unsqueeze(audio, 0)
    model.eval()
    model = model.to("cpu")
    with torch.no_grad():
        pred = model(audio)
        pred = torch.squeeze(audio)
        pred = audio.numpy()
    pred = int_list_to_ipa(pred)
    print(pred)
