from catattack import data

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

train_set, valid_set = data.get_translation_dataset(
    reverse=False,
    src_valid_prefixes=eng_prefixes,
    sentence_length=None,  # We will use a single sentence per training loop
    split_ratio=0.8)

model = model.get_rnn_seq2seq()
