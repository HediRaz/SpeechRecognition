import torch
import matplotlib.pyplot as plt
import librosa
from Utils.utils_dataset import int_list_to_ipa


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)


def decoder(pred):
    pred = torch.softmax(pred, 1)
    pred = torch.argmax(pred, 1)
    pred = torch.unique_consecutive(pred)
    pred = pred.to("cpu").numpy()
    pred = int_list_to_ipa(pred)
    return pred

