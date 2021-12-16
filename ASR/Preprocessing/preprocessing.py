import os

import eng_to_ipa
import numpy as np
import torch
import torchaudio
from Utils import utils_dataset


def spectogram_pipline(datapath,  window_size=25, window_step=20, resample_rate=16000):
    """
    window size = 25 ms
    window step = 20 ms
     
    window type = hann function
    """

    data, sample_rate = utils_dataset.load_audio_file(datapath, resample_rate=resample_rate)
    window_size = int(window_size * 1e-3 * sample_rate)
    window_step = int(window_step * 1e-3 * sample_rate)
    nfft = window_size

    data = torch.squeeze(data)
    stfts = torchaudio.functional.spectrogram(data, window=torch.hann_window(window_size), n_fft=nfft, hop_length=window_step, win_length=window_size, pad=0, power=2, normalized=True, return_complex=False)

    return stfts


def mel_pipeline(datapath, window_size, window_step, normalized, resample_rate=16000):

    data, sample_rate = utils_dataset.load_audio_file(datapath, resample_rate=resample_rate)
    window_size = int(window_size * 1e-3 * sample_rate)
    window_step = int(window_step * 1e-3 * sample_rate)
    nfft = window_size

    data = torch.squeeze(data)
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=128)
    return mel_transform(data)


def phonem_pipeline(label):
    label = eng_to_ipa.convert(label)
    label = utils_dataset.ipa_to_int_list(label)
    label = np.array(label, dtype=np.uint8)
    return label


def char_pipeline(label):
    label = utils_dataset.char_to_int_list(label)
    label = np.array(label, dtype=np.uint8)
    return label


def preproces_dataset(raw_ds_folder):
    raw_ds_folder = os.path.normpath(raw_ds_folder)
    processed_ds_folder = raw_ds_folder+"-processed"
    utils_dataset.create_folder_if_not_exist(processed_ds_folder)
    for audio_filename, label, id in utils_dataset.yield_label_and_audio_filenames_from_raw_dataset(raw_ds_folder):
        # data id
        id_person, id_chapter, id_audio = id.split("-")

        # Create folders
        person_path = os.path.join(processed_ds_folder, id_person)
        utils_dataset.create_folder_if_not_exist(person_path)
        chapter_path = os.path.join(person_path, id_chapter)
        utils_dataset.create_folder_if_not_exist(chapter_path)

        # process data
        mel_spectogram = spectogram_pipline(audio_filename)
        # label = char_pipeline(label)
        label = phonem_pipeline(label)

        # save processed data
        spec_filename = f"{id_person}-{id_chapter}-{id_audio}-audio.pt"
        torch.save(mel_spectogram, os.path.join(chapter_path, spec_filename))
        label_filename = f"{id_person}-{id_chapter}-{id_audio}-label.npy"
        np.save(os.path.join(chapter_path, label_filename), label)



if __name__ == "__main__":
    preproces_dataset("Datasets/LibriSpeech/dev-clean")
