import torch
import torch.nn as nn
from vocoders import hifigan
from vocoders.vocgan import vocgan_generator
import json
import numpy as np

from utils.hparams import hparams as hps
from English.tacotron2 import Tacotron2 as EnTacotron2
from Korean.tacotron2 import Tacotron2 as KoTacotron2
from utils.util import get_mask_from_lengths


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.loss = nn.MSELoss(reduction = 'none')

    def forward(self, model_outputs, targets):
        mel_out, mel_out_postnet, gate_out, _ = model_outputs
        gate_out = gate_out.view(-1, 1)

        mel_target, gate_target, output_lengths = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        output_lengths.requires_grad = False
        slice = torch.arange(0, gate_target.size(1), hps.n_frames_per_step)
        gate_target = gate_target[:, slice].view(-1, 1)
        mel_mask = ~get_mask_from_lengths(output_lengths.data, True)

        mel_loss = self.loss(mel_out, mel_target) + \
            self.loss(mel_out_postnet, mel_target)
        mel_loss = mel_loss.sum(1).masked_fill_(mel_mask, 0.)/mel_loss.size(1)
        mel_loss = mel_loss.sum()/output_lengths.sum()

        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss+gate_loss, (mel_loss.item(), gate_loss.item())



def get_tacotron2(language: str):
    lang = language.lower()
    if lang == 'english' or lang == 'en':
        return EnTacotron2()
    elif lang == 'korean' or lang == 'ko':
        return KoTacotron2()


def get_vocoder(name, device):
    if name == 'MelGAN':
        vocoder = torch.hub.load("descriptinc/melgan-neurips", "load_melgan", "linda_johnson")

        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)

    elif name == "HiFi-GAN":
        with open("vocoders/hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)

        ckpt = torch.load("vocoders/hifigan/generator_LJSpeech.pth", map_location=device)
        vocoder.load_state_dict(ckpt['generator'])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    elif name == 'vocgan':
        vocoder = vocgan_generator.Generator(80, 4)
        ckpt = torch.load("vocoders/vocgan/vocgan_kss_pretrained_model_epoch_4500.pt", map_location=device)
        vocoder.load_state_dict(ckpt['model_g'])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, name, lengths=None):
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)
        elif name == "vocgan":
            if len(mels.shape) == 2:
                mels = mels.unsqueeze(0)
            wavs = vocoder.infer(mels).squeeze(0)

    wavs = (
        wavs.cpu().numpy()
        * 32768.0
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs