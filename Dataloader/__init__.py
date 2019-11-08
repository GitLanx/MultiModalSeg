from .A import ALoader
from .B import BLoader
from .irseg_loader import IRSegLoader

VALID_DATASET = ['a', 'b', 'irseg']


def get_loader(dataset):
    if dataset.lower() == 'a':
        return ALoader
    elif dataset.lower() == 'b':
        return BLoader
    elif dataset.lower() == 'irseg':
        return IRSegLoader
    else:
        raise ValueError('Unsupported dataset, '
                         'valid datasets as follows:\n{}\n'.format(', '.join(VALID_DATASET)))
