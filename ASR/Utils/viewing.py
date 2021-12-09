import torch
import matplotlib.pyplot as plt
import librosa
from utils_dataset import SoundDataset
from utils_dataset import create_dataloaders


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show()



ds = SoundDataset("Datasets/LibriSpeech/dev-clean-processed")
# train_dl, test_dl = create_dataloaders(ds, batch_size=1, split=0.8)
for i in range(10):
    x, y, xs, ys = ds[i]
    print(xs, ys)
