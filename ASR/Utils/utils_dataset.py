import os
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def load_audio_file(filename, resample_rate=None):
    """ 
    Load an audio file

    Args:
    filename : path of the audio file
    resample_rate (Optional) : freq we want to resample the audio

    Returns:
    waveform (Tensor): the audio
    sample_rate (int): the sample rate

    """
    waveform, sample_rate = torchaudio.load(filename)
    if resample_rate is not None:
        waveform = F.resample(waveform, sample_rate, resample_rate, resampling_method="kaiser_window")
        sample_rate = resample_rate
    return waveform, sample_rate


def yield_label_and_audio_filenames_from_raw_dataset(raw_ds_folder):
    """
    Iterate over the LibriSpeech dataset and yield the audio and label path

    Args;
    folder : folder path of a folder in LibriSpeech dataset

    yield: audio_filename, raw label (str), id (str)

    """
    # Get reader's and chapter's id
    raw_ds_folder = os.path.normpath(raw_ds_folder)
    for id_person in os.listdir(raw_ds_folder):
        person_path = os.path.join(raw_ds_folder, id_person)
        for id_chapter in os.listdir(person_path):
            chapter_path = os.path.join(person_path, id_chapter)

            # Iterate over the folder
            labels_filename = os.path.join(chapter_path, id_person+"-"+id_chapter+".trans.txt")
            with open(labels_filename, 'r') as labels:
                for line in labels.readlines():
                    length_id = len(line.split(' ')[0])
                    label = line[length_id+1:-1]  # delete id and \n

                    audio_filename = os.path.join(chapter_path, line[:length_id]+".flac")
                    yield audio_filename, label, line[:length_id]


def ipa_to_int_list(text):
    """ Convert a chain of ipa character into list of int """
    dict_ipa_to_int = {
        ' ': 0,
        's': 1,
        'w': 2,
        '??': 3,
        '??': 4,
        '??': 5,
        '??': 6,
        'a': 7,
        'r': 8,
        'p': 9,
        'n': 10,
        "'": 11,
        'z': 12,
        'l': 13,
        '??': 14,
        '??': 15,
        'h': 16,
        '*': 17,
        'o': 18,
        '??': 19,
        '??': 20,
        'd': 21,
        'k': 22,
        '??': 23,
        'e': 24,
        'm': 25,
        'u': 26,
        '??': 27,
        '??': 28,
        'q': 29,
        '??': 30,
        '??': 31,
        'f': 32,
        'c': 33,
        'i': 34,
        'j': 35,
        '??': 36,
        '??': 37,
        'v': 38,
        'y': 39,
        'b': 40,
        'g': 41,
        '??': 42,
        'x': 43,
        't': 44
    }
    return [dict_ipa_to_int[c] for c in text]


def int_list_to_ipa(l):
    """ Convert a chain of int between 0 and 45 into list of ipa character """
    dict_ipa_to_int = {
        ' ': 0,
        's': 1,
        'w': 2,
        '??': 3,
        '??': 4,
        '??': 5,
        '??': 6,
        'a': 7,
        'r': 8,
        'p': 9,
        'n': 10,
        "'": 11,
        'z': 12,
        'l': 13,
        '??': 14,
        '??': 15,
        'h': 16,
        '*': 17,
        'o': 18,
        '??': 19,
        '??': 20,
        'd': 21,
        'k': 22,
        '??': 23,
        'e': 24,
        'm': 25,
        'u': 26,
        '??': 27,
        '??': 28,
        'q': 29,
        '??': 30,
        '??': 31,
        'f': 32,
        'c': 33,
        'i': 34,
        'j': 35,
        '??': 36,
        '??': 37,
        'v': 38,
        'y': 39,
        'b': 40,
        'g': 41,
        '??': 42,
        'x': 43,
        't': 44,
        '': 45
    }
    dict_int_to_ipa = dict([(a, b) for (b, a) in dict_ipa_to_int.items()])
    return "".join([dict_int_to_ipa[c] for c in l])


def char_to_int_list(text):
    """ Convert a chain of character into list of int """
    d = {
        " ": 0,
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7,
        "H": 8,
        "I": 9,
        "J": 10,
        "K": 11,
        "L": 12,
        "M": 13,
        "N": 14,
        "O": 15,
        "P": 16,
        "Q": 17,
        "R": 18,
        "S": 19,
        "T": 20,
        "U": 21,
        "V": 22,
        "W": 23,
        "X": 24,
        "Y": 25,
        "Z": 26,
        "'": 27,
        "": 28,
    }
    return [d[c] for c in text]


def int_list_to_char(l):
    """ Convert a chain of int between 0 and 28 into list of ipa character """
    d = {
        " ": 0,
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7,
        "H": 8,
        "I": 9,
        "J": 10,
        "K": 11,
        "L": 12,
        "M": 13,
        "N": 14,
        "O": 15,
        "P": 16,
        "Q": 17,
        "R": 18,
        "S": 19,
        "T": 20,
        "U": 21,
        "V": 22,
        "W": 23,
        "X": 24,
        "Y": 25,
        "Z": 26,
        "'": 27,
        "": 28,
    }
    d = dict([(a, b) for (b, a) in d.items()])
    return "".join([d[c] for c in l])


def create_folder_if_not_exist(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def get_all_id(ds_folder):
    """ Get all id of a processed dataset """
    all_id = []
    ds_folder = os.path.normpath(ds_folder)
    for id_person in os.listdir(ds_folder):
        path_person = os.path.join(ds_folder, id_person)
        for id_chapter in os.listdir(path_person):
            path_chapter = os.path.join(path_person, id_chapter)
            n = len(os.listdir(path_chapter))//2
            all_id.extend([f"{id_person}-{id_chapter}-{i:>04}" for i in range(n)])
    return all_id


def id_to_paths(ds_folder, id):
    """ Given an id, return the path of the audio file and the label """
    ds_folder = os.path.normpath(ds_folder)
    id_person, id_chapter, id_audio = id.split("-")
    audio_path = os.path.join(ds_folder, id_person, id_chapter, f"{id}-audio.pt")
    label_path = os.path.join(ds_folder, id_person, id_chapter, f"{id}-label.npy")
    return audio_path, label_path


def load_all_dataset(ds_folder):
    """ Get all audio and label path """
    ds_folder = os.path.normpath(ds_folder)
    all_id = get_all_id(ds_folder)
    all_paths = list(map(lambda x: id_to_paths(ds_folder, x), all_id))
    return all_paths


class SoundDataset(Dataset):

    def __init__(self, ds_folder):
        super().__init__()
        ds_folder = os.path.normpath(ds_folder)
        self.ds_folder = ds_folder
        self.all_paths = load_all_dataset(ds_folder)

        self.training = False
        self.freq_masking = T.FrequencyMasking(15)
        self.time_masking = T.TimeMasking(35)
        self.time_stretch = T.TimeStretch()

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        audio = torch.load(self.all_paths[idx][0])
        label = np.load(self.all_paths[idx][1])
        label = torch.tensor(label, dtype=torch.int64)
        xs, ys = audio.size(1), label.size(0)

        # Spec Augmentation
        if self.training:
            audio = self.freq_masking(audio)
            audio = self.time_masking(audio)

        return audio, label, xs, ys


def collate_fn(batch):
    """ Pad the batch """
    batch_audio = [item[0] for item in batch]
    batch_label = [item[1] for item in batch]
    batch_xs = [item[2] for item in batch]
    batch_ys = [item[3] for item in batch]

    batch_audio = torch.nn.utils.rnn.pad_sequence([audio.transpose(0, 1) for audio in batch_audio], batch_first=True).transpose(1, 2).unsqueeze(1)
    batch_label = torch.nn.utils.rnn.pad_sequence([label for label in batch_label], batch_first=True)
    batch_xs = torch.tensor(batch_xs, dtype=torch.int64)
    batch_ys = torch.tensor(batch_ys, dtype=torch.int64)
    return batch_audio, batch_label, batch_xs, batch_ys


def create_dataloaders(ds, batch_size=16, split=0.8):
    n = len(ds)
    n_train = int(split * n)
    train_ds, val_ds = random_split(ds, [n_train, n - n_train])

    train_ds.training = True
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)

    return train_dl, val_dl
