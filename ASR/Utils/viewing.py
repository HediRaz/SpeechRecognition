import torch
import torchmetrics
import matplotlib.pyplot as plt
import librosa
from Utils.utils_dataset import int_list_to_char
from Utils.utils_dataset import int_list_to_ipa


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    """ Visualization of a spectrogram """
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)


def greedy_decoder(preds):
    """ Decode model prediction into a sentence """
    preds = torch.softmax(preds, -1)
    preds = torch.argmax(preds, -1)
    preds = [torch.unique_consecutive(p) for p in preds]
    preds = [p.to("cpu").numpy() for p in preds]
    preds = [int_list_to_ipa(p) for p in preds]
    return preds


class Metric():
    """ Class to name the metrics """
    def __init__(self, name="not named"):
        self.name = name
    
    def __call__(preds, labels):
        pass


class BertScoreMetric(Metric):
    def __init__(self):
        super().__init__(name="Bert Score")
    
    def __call__(self, preds, labels):
        return torchmetrics.functional.bert_score(
            predictions=preds,
            references=labels,
            lang="en"
        )


class CERMetric(Metric):
    def __init__(self):
        super().__init__(name="CER")
    
    def __call__(self, preds, labels):
        return torchmetrics.functional.char_error_rate(predictions=preds, references=labels)


class WERMetric(Metric):
    def __init__(self):
        super().__init__(name="WER")
    
    def __call__(self, preds, labels):
        return torchmetrics.functional.wer(predictions=preds, references=labels)


class MetricsPrint():
    """ Class to visualize metrics during training """
    def __init__(self, metrics):
        self.metrics_names = [m.name for m in metrics]        

    def initial_print(self, total_batch, name=" "*16):
        self.total_batch = total_batch
        text = name
        text += "|    Loss    "  # 13 characters
        for name in self.metrics_names:
            n = 12 - len(name)
            text += "|" + " " * (n//2) + name + " " * (n//2 + n%2)
        print(text)

    def print_loss_metrics(self, loss_value, metrics_values, batch):
        text = f"{batch:>6} / {self.total_batch:>6} "
        text += f"|  {loss_value:.6f}  "
        for v in metrics_values:
            text += f"|  {v:.6f}  "
        print(text)
