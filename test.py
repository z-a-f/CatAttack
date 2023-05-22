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
    sentence_length=10,
    split_ratio=0.8)

print(len(train_set))
