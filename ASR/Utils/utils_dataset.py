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
    waveform, sample_rate = torchaudio.load(filename)
    if resample_rate is not None:
        waveform = F.resample(waveform, sample_rate, resample_rate, resampling_method="kaiser_window")
        sample_rate = resample_rate
    return waveform, sample_rate


def yield_label_and_audio_filenames_from_raw_dataset(raw_ds_folder):
    """
    folder : folder path of a folder in LibriSpeech dataset

    yield: audio_filename, raw label: str, id: str

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


def pad_label(label, size):
    if label.size(0) >= size:
        label = label[:size]

    else:
        label = pad(label, (0, size - label.size(0)), "constant", 0)

    return label


def pad_spectogram(spec, size):
    if spec.size(1) >= size:
        spec = spec[:, :size]

    else:
        spec = pad(spec, (0, size - spec.size(1), 0, 0), "constant", 0)
    return spec


def ipa_to_int_list(text):
    dict_ipa_to_int = {
        ' ': 0,
        's': 1,
        'w': 2,
        'ˌ': 3,
        'θ': 4,
        'ɑ': 5,
        'ŋ': 6,
        'a': 7,
        'r': 8,
        'p': 9,
        'n': 10,
        "'": 11,
        'z': 12,
        'l': 13,
        'ʊ': 14,
        'ɪ': 15,
        'h': 16,
        '*': 17,
        'o': 18,
        'ˈ': 19,
        'ʧ': 20,
        'd': 21,
        'k': 22,
        'ð': 23,
        'e': 24,
        'm': 25,
        'u': 26,
        'ʤ': 27,
        'ə': 28,
        'q': 29,
        'æ': 30,
        'ʃ': 31,
        'f': 32,
        'c': 33,
        'i': 34,
        'j': 35,
        'ʒ': 36,
        'ɛ': 37,
        'v': 38,
        'y': 39,
        'b': 40,
        'g': 41,
        'ɔ': 42,
        'x': 43,
        't': 44
    }
    return [dict_ipa_to_int[c] for c in text]


def int_list_to_ipa(l):
    dict_ipa_to_int = {
        ' ': 0,
        's': 1,
        'w': 2,
        'ˌ': 3,
        'θ': 4,
        'ɑ': 5,
        'ŋ': 6,
        'a': 7,
        'r': 8,
        'p': 9,
        'n': 10,
        "'": 11,
        'z': 12,
        'l': 13,
        'ʊ': 14,
        'ɪ': 15,
        'h': 16,
        '*': 17,
        'o': 18,
        'ˈ': 19,
        'ʧ': 20,
        'd': 21,
        'k': 22,
        'ð': 23,
        'e': 24,
        'm': 25,
        'u': 26,
        'ʤ': 27,
        'ə': 28,
        'q': 29,
        'æ': 30,
        'ʃ': 31,
        'f': 32,
        'c': 33,
        'i': 34,
        'j': 35,
        'ʒ': 36,
        'ɛ': 37,
        'v': 38,
        'y': 39,
        'b': 40,
        'g': 41,
        'ɔ': 42,
        'x': 43,
        't': 44
    }
    dict_int_to_ipa = dict([(a, b) for (b, a) in dict_ipa_to_int.items()])
    return "".join([dict_int_to_ipa[c] for c in l])


def create_folder_if_not_exist(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def get_all_id(ds_folder):
    """
    Get all id of a processed dataset

    """
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
    ds_folder = os.path.normpath(ds_folder)
    id_person, id_chapter, id_audio = id.split("-")
    audio_path = os.path.join(ds_folder, id_person, id_chapter, f"{id}-audio.pt")
    label_path = os.path.join(ds_folder, id_person, id_chapter, f"{id}-label.npy")
    return audio_path, label_path


def load_all_dataset(ds_folder):
    ds_folder = os.path.normpath(ds_folder)
    all_id = get_all_id(ds_folder)
    all_paths = list(map(lambda x: id_to_paths(ds_folder, x), all_id))
    all_audio = [torch.load(path[0]) for path in all_paths]
    all_label = [torch.tensor(np.load(path[1]), dtype=torch.int64) for path in all_paths]

    # Get data size
    all_audio_size = torch.tensor([l.size(-1) for l in all_audio], dtype=torch.long, device=device)
    all_label_size = torch.tensor([l.size(0) for l in all_label], dtype=torch.long, device=device)

    # Pad data
    max_shape_audio = max(all_audio, key=lambda x: x.shape[1]).shape[1]
    max_shape_label = max(all_label, key=lambda x: x.shape[0]).shape[0]
    all_audio = [pad_spectogram(a, max_shape_audio) for a in all_audio]
    all_label = [pad_label(a, max_shape_label) for a in all_label]

    # Create tensor
    all_audio = torch.stack(all_audio, 0)
    all_label = torch.stack(all_label, 0)
    all_audio = all_audio.to(device)
    all_label = all_label.to(device)

    return all_audio, all_label, all_audio_size, all_label_size


class SoundDataset(Dataset):

    def __init__(self, ds_folder):
        super().__init__()
        ds_folder = os.path.normpath(ds_folder)
        self.ds_folder = ds_folder
        self.all_audio, self.all_label, self.all_audio_size, self.all_label_size = load_all_dataset(ds_folder)

        self.training = False
        self.freq_masking = T.FrequencyMasking(20)
        self.time_masking = T.TimeMasking(20)
        self.time_stretch = T.TimeStretch()

    def __len__(self):
        return self.all_label.shape[0]

    def __getitem__(self, idx):
        audio = self.all_audio[idx]
        if self.training:
            audio = self.freq_masking(audio)
            audio = self.time_masking(audio)
            rate = np.random.uniform(.9, 1.1)
            audio = self.time_masking(audio, rate)
        return audio, self.all_label[idx], self.all_audio_size[idx], self.all_label_size[idx]


def create_dataloaders(ds, batch_size=16, split=0.8):
    n = len(ds)
    n_train = int(split * n)
    train_ds, val_ds = random_split(ds, [n_train, n - n_train])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    train_dl.training = True
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return train_dl, val_dl



if __name__ == "__main__":
    ds = SoundDataset("Datasets/LibriSpeech/dev-clean-processed")
    train_dl, test_dl = create_dataloaders(ds)
    x = next(train_dl._get_iterator())
    print(x)
    print(x[0].shape)
    print(x[1].shape)
    print(x[2].shape)
