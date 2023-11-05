from utils.hparams import hparams
from Korean.text import symbols


class ko_hparams(hparams):
    seed = 1234

    ################################
    # Data Parameters              #
    ################################
    text_cleaners = ['korean_cleaners']
    cleaners = 'korean_cleaners',

    ################################
    # Train info                   #
    ################################
    n_symbols = len(symbols.symbols)
    eg_text = '타코트론은 강력한 생성기이다.'
