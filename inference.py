import os
import sys

import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
from Korean.text import text_to_sequence as t2s_ko
from English.text import text_to_sequence as t2s_en
from model.builder import get_tacotron2, get_vocoder, vocoder_infer

from Korean.ko_hparams import ko_hparams
from English.en_hparams import en_hparams

from utils.util import mode, to_arr
from scipy.io import wavfile
from English.audio import save_wav, inv_melspectrogram


def load_model(ckpt_pth, language, device):
    ckpt_dict = torch.load(ckpt_pth, map_location=device)
    model = get_tacotron2(language)
    model.load_state_dict(ckpt_dict['model'])
    model = mode(model, True).eval()
    return model


def infer(text, model, language):
    if language == 'english' or language == 'en':
        sequence = t2s_en(text, hps.text_cleaners)
    else:
        sequence = t2s_ko(text)
    sequence = mode(torch.IntTensor(sequence)[None, :]).long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    return (mel_outputs, mel_outputs_postnet, alignments)


def plot_data(data, figsize = (16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize = figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect = 'auto', origin = 'bottom')

def audio(output, pth):
    mel_outputs, mel_outputs_postnet, _ = output
    wav_postnet = inv_melspectrogram(to_arr(mel_outputs_postnet[0]))
    save_wav(wav_postnet, pth+'.wav')


def plot(output, pth):
    mel_outputs, mel_outputs_postnet, alignments = output
    plot_data((to_arr(mel_outputs[0]),
                to_arr(mel_outputs_postnet[0]),
                to_arr(alignments[0]).T))
    plt.savefig(pth+'.png')


if __name__ == '__main__':
    # sys ['inference.py', 'korean', 'user_root', 'ckp_path', 'ckp_pth', 'best_ckp_name', nex]
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--language', type=str, default='english',
                        help='TTS langauge [korean(or ko), english(or en)]')
    parser.add_argument('--ckp_path', type=str, default='None',
                        help='유저 목소리로 학습된 해당 언어 TTS 모델 checkpoint 가 저장된 경로')
    parser.add_argument('--best_ckp_name', type=str, default='None.pth.tar',
                        help='학습된 해당 언어의 best score checkpoint 의 이름')
    parser.add_argument('--output_path', type=str, default='./infer',
                        help='최종 output이 저장되는 경로')
    parser.add_argument('--output_name', type=str, default='infer.wav',
                        help='최종 output 이름')
    parser.add_argument('-t', '--text', type=str, default='The case against the founder of the failed FTX exchange had come to symbolize the excesses of the volatile cryptocurrency industry.',
                        help='text to synthesize')

    args = parser.parse_args()

    sample_ckp_path = None
    lang = args.language.lower()
    # print(lang)
    if lang == 'english' or lang == 'en':
        hps = en_hparams
        vocoer_name = 'HiFi-GAN'
        sample_ckp_path = 'sample_voice_ckpt/en_female'

    elif lang == 'korean' or lang == 'ko':
        hps = ko_hparams
        vocoer_name = 'vocgan'
        sample_ckp_path = 'sample_voice_ckpt/ko_female.pth.tar'

    else:
        exit('한국어와 영어만 지원합니다.\n사용가능한 language --> korean(or ko) and english(or en) [대소문자 상관 없음]')

    device = 'cpu'
    if hps.is_cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        device = 'cuda'

    ckp_path = f'{args.ckp_path}/{args.best_ckp_name}'
    if os.path.exists(ckp_path):
        model = load_model(ckp_path, args.language, device)
        model_dynamic_quantized = torch.quantization.quantize_dynamic(
            model, qconfig_spec={torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8
        )
        output = infer(args.text, model_dynamic_quantized, args.language)
    else:
        model = load_model(sample_ckp_path, args.language, device)
        model_dynamic_quantized = torch.quantization.quantize_dynamic(
            model, qconfig_spec={torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8
        )
        output = infer(args.text, model_dynamic_quantized, args.language)

    vocoder = get_vocoder(vocoer_name, device)
    vocoder_dynamic_quantized = torch.quantization.quantize_dynamic(
        vocoder, qconfig_spec={torch.nn.Linear, torch.nn.ConvTranspose1d, torch.nn.Conv1d}, dtype=torch.qint8
    )
    wav = vocoder_infer(output[1], vocoder_dynamic_quantized, vocoer_name)

    # output path가 존재하지 않으면 생성
    os.makedirs(args.output_path, exist_ok=True)
    path = f'{args.output_path}/{args.output_name}'
    wavfile.write(path, hps.sample_rate, wav[0])

    print("Complete Inference !!!")

    sys.exit()