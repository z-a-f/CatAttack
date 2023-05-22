# Cat Attack

```
 /\_/\
( o.o )
 > ^ <
```

Tutorials based on https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Notes:

* There are a lot of custom modifications relative to the original tutorial, so pay attention

## Architecture

```
"le chat est noir <EOS>"                "<SOS> the cat is black"
       |    |    |                            |    |    |
       v    v    v                            v    v    v
  |>---------------->|   Context Vector   |------------------|
  |>    Encoder     >|------------------->|     Decoder      |
  |>---------------->|                    |------------------|
                                              |    |    |
                                              v    v    v
                                         "the cat is black <EOS>"
```


# Dataset and Tokenizer

```python
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

src_tokenizer = train_set.src_tokenizer
dst_tokenizer = train_set.dst_tokenizer
```

# Models

## RNN Seq2Seq

```python

```
