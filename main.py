import os

from catattack import data
from catattack import model

CACHE_PATH = '.cache'

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

PROCESSED_CACHE_PATH = os.path.join(CACHE_PATH, 'processed')
os.makedirs(PROCESSED_CACHE_PATH, exist_ok=True)
full_set_path = os.path.join(PROCESSED_CACHE_PATH, 'translation_set.zpkl')
train_set_path = os.path.join(PROCESSED_CACHE_PATH, 'train_set.zpkl')
valid_set_path = os.path.join(PROCESSED_CACHE_PATH, 'valid_set.zpkl')

# Load the files

if not os.path.isfile(train_set_path) or not os.path.isfile(valid_set_path):
    # If the train and valid set don't exist
    if os.path.isfile(full_set_path):
        translation_dataset = data.get_translation_dataset(load_from=full_set_path)
    else:
        translation_dataset = data.get_translation_dataset(
            reverse=False,
            src_valid_prefixes=eng_prefixes,
            sentence_length=None,  # We will use a single sentence per training loop
            cache_dir=CACHE_PATH)
        translation_dataset.save(full_set_path)
    train_set, valid_set = translation_dataset.split(0.8)
    train_set.save(train_set_path)
    valid_set.save(valid_set_path)
else:
    train_set = data.get_translation_dataset(load_from=train_set_path)
    valid_set = data.get_translation_dataset(load_from=valid_set_path)

# rnn_model = model.get_rnn_seq2seq()
