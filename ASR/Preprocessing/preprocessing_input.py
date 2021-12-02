import torch
from torch.autograd.grad_mode import F
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os


def spectogram_pipline(datapath,  window_size=25, window_step=20):
    """
    window size = 25 ms
    window step = 20 ms
     
    window type = hann function
    """

    data, sample_rate = torchaudio.load(datapath)
    window_size = int(window_size * 1e-3 * sample_rate)
    window_step = int(window_step * 1e-3 * sample_rate)
    nfft = window_size

    data = torch.squeeze(data)
    stfts = torchaudio.functional.spectrogram(data, window=torch.hann_window(window_size), n_fft=nfft, hop_length=window_step, win_length=window_size, pad=0, power=2, normalized=True, return_complex=False)

    return stfts

def mel_pipeline(datapath, window_size, window_step, normalized):

    data, sample_rate = torchaudio.load(datapath)
    window_size = int(window_size * 1e-3 * sample_rate)
    window_step = int(window_step * 1e-3 * sample_rate)
    nfft = window_size

    data = torch.squeeze(data)
    mel_spectgram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=nfft, win_length=window_size, hop_lengtt=window_step, power=2.0, normalized=normalized)
    return mel_spectgram

if __name__ == "__main__":

    path = r"LibriSpeech\dev-clean\84\121123"
    list_file = os.listdir(path)
    print(list_file)
    for file in list_file[:-1]:
        plt.figure()
        mel_specgram = mel_pipeline(r"LibriSpeech\dev-clean\84\121123\\" + file, 25, 20,False)
        print(mel_specgram.shape)
        plt.imshow(mel_specgram)
    plt.show()