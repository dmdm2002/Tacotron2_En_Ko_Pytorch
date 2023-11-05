from utils.hparams import hparams
from English.text import symbols


class en_hparams(hparams):
    seed = 1234

    ################################
    # Data Parameters              #
    ################################
    text_cleaners = ['english_cleaners']

    ################################
    # Audio and Train info         #
    ################################=
    n_symbols = len(symbols)
    eg_text = 'OMAK is a thinking process which considers things always positively.'
