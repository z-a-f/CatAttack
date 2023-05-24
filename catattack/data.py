
__all__ = [
    'get_data',
    'get_translation_dataset',
    'TranslationDataset',
]


from .tokenizer import Lang

from ._data import downloader
from ._data import loader
from ._data import dataset

from ._data.dataset import TranslationDataset


def get_data(name: str, cache_dir='.cache'):
    data = None
    if name == 'pytorch_tutorial_text':
        data_path = downloader._get_pytorch_tutorial_text(cache_dir)
        data_pairs = loader._load_pytorch_tutorial_text(data_path)
        eng, fra = list(zip(*data_pairs))
        data = {
            'eng': eng,
            'fra': fra
        }
    else:
        raise ValueError(f"I don't know dataset {name}. Please choose from {downloader.URLS.keys()}")
    return data

def get_translation_dataset(*,
                            load_from=None,
                            reverse=False,
                            src_max_length=None, dst_max_length=None,
                            src_valid_prefixes=None, dst_valid_prefixes=None,
                            sentence_length=None,
                            split_ratio=None,
                            cache_dir='.cache'):
    if load_from is not None:
        print(f'===> INFO: Loading the translation dataset from {load_from}; ignoring all arguments.')
        return dataset.TranslationDataset.load(load_from)

    sentences = get_data('pytorch_tutorial_text', cache_dir=cache_dir)

    src_name = 'eng'
    dst_name = 'fra'
    if reverse:
        src_name, dst_name = dst_name, src_name
    src = sentences[src_name]
    dst = sentences[dst_name]
    src_tokenizer = Lang(name=src_name)
    dst_tokenizer = Lang(name=dst_name)

    translation_dataset = TranslationDataset(
        src=src, dst=dst,  # Assume that src[idx] and dst[idx] are the same thing
        src_name=src_name, dst_name=dst_name,
        src_max_length=src_max_length, dst_max_length=dst_max_length,
        src_valid_prefixes=src_valid_prefixes, dst_valid_prefixes=dst_valid_prefixes,
        src_tokenizer=src_tokenizer, dst_tokenizer=dst_tokenizer,
        shuffle=True,
        sentence_length=sentence_length,
    )

    if split_ratio is not None:
        return translation_dataset.split(split_ratio)
    else:
        return translation_dataset
