# Tacotron2_En_Ko_Pytorch
## Code
- Tacotron2 model 과 HiFi-GAN(영어), vocgan(한국어) 을 결합하여 한국어와 영어 TTS 를 구현하는 project 입니다.
- 본 코드의 원본은 아래와 같습니다.
  - https://github.com/hccho2/Tacotron2-Wavenet-Korean-TTS
  - https://github.com/BogiHsu/Tacotron2-PyTorch

## Pretrained weight
### Tacotreon2 의 pretrained weight 출처
- 영어
  - LJ Speech dataset 으로 학습: https://github.com/BogiHsu/Tacotron2-PyTorch
### 영어와 한국어의 vocoder pretrained weight 출처
  - 영어: https://github.com/ming024/FastSpeech2
  - 한국어: https://github.com/HGU-DLLAB/Korean-FastSpeech2-Pytorch

## 직접 학습
- 영어-CMU_ARCTIC datasets
  - LJ Speech dataset pretrained weight를 사용하여 cmu_us_bdl_arctic dataset 으로 finetuning 진행 100k 학습
- 한국어
  - KSS dataset 으로 직접 학습을 진행 100k 학습
