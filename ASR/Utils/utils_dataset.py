import os
import torchaudio
import torchaudio.functional as F
import eng_to_ipa
from torch.nn.functional import pad


def load_file(filename, resample_rate=None):
    waveform, sample_rate = torchaudio.load(filename)
    if resample_rate is not None:
        waveform = F.resample(waveform, sample_rate, resample_rate, resampling_method="kaiser_window")
        sample_rate = resample_rate
    return waveform[0], sample_rate


def load_folder(folder, resample_rate=None, to_ipa=False):
    """
    folder : folder path of a folder in LibriSpeech dataset

    """
    # Get reader's and chapter's id
    folder = os.path.normpath(folder)
    h, id_chapter = os.path.split(folder)
    if len(id_chapter) == 0:
        h, id_chapter = os.path.split(h)
    h, id_person = os.path.split(h)

    # Iterate over the folder
    labels_filename = os.path.join(h, id_person, id_chapter, id_person+"-"+id_chapter+".trans.txt")
    with open(labels_filename, 'r') as labels:
        for line in labels.readlines():
            length_id = len(line.split(' ')[0])
            label = line[length_id+1:-1]  # delete id and \n
            if to_ipa:
                label = eng_to_ipa.convert(label)

            audio_filename = os.path.join(h, id_person, id_chapter, line[:length_id]+".flac")
            waveform, sample_rate = load_file(audio_filename, resample_rate=resample_rate)
            yield waveform, label


def resize_waveform(waveform, size):
    if waveform.size(1) >= size:
        waveform = waveform[:, :size]

    else:
        waveform = pad(waveform, (0, size - waveform.size(1)), "constant", 0)

    return waveform


def ipa_to_int_tensor(text):
    from tqdm import tqdm
    all_symbols = set()
    for id_reader in tqdm(os.listdir("Datasets/LibriSpeech/dev-clean")):
        for id_chapter in os.listdir(f"Datasets/LibriSpeech/dev-clean/{id_reader}"):
            for _, l in load_folder(f"Datasets/LibriSpeech/dev-clean/{id_reader}/{id_chapter}", to_ipa=True):
                for c in l:
                    all_symbols.add(c)
    
    print(all_symbols)

ipa_to_int_tensor(1)
