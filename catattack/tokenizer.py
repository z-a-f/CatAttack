from joblib import Parallel, delayed
from collections import Counter

from torchtext.vocab import Vocab

from ._data import utils


class Language:
    def __init__(self, *,
                 sos_token=None,
                 eos_token=None,
                 pad_token=None,
                 unk_token=None,
                 lower=True,
                 include_lengths=True,
                ):
        self.sos_token = sos_token or '<SOS>'
        self.eos_token = eos_token or '<EOS>'
        self.pad_token = pad_token or '<PAD>'
        self.unk_token = unk_token or '<UNK>'
        self.lower = lower
        self.include_lengths = include_lengths

        self.vocab = Vocabulary()

    def build_vocab(self, data, min_freq=None):
        pass


class Lang:
    pass
#     r"""Language word tokenizer.

#     """
#     def __init__(self, name):
#         self.name = name
#         self.word_count = Counter()  # Frequency counter
#         self.word2idx = {}
#         self.idx2word = []

#         self.reserved_idx2word = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']
#         self.reserved_word2idx = {word: idx for idx, word in enumerate(self.reserved_idx2word)}

#         self.debug = False

#     def add(self, sentences):
#         all_sentences = self.cleanup(sentences)
#         all_sentences = (' '.join(sentences)).split()
#         self.word_count.update(Counter(all_sentences))
#         return self

#     def make_tokens(self, max_tokens=None):
#         sorted_words = sorted(self.word_count.keys(), key=lambda w: self.word_count[w], reverse=True)
#         if max_tokens:
#             sorted_words = sorted_words[:max_tokens]
#         self.idx2word = self.reserved_idx2word + sorted_words
#         self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
#         return self

#     def __len__(self):
#         return len(self.idx2word)

#     def tokenize(self, sentences: list, sentence_length=None, add_sos=False, add_eos=False):
#         # Assume input is a list
#         result = Parallel(n_jobs=-1, verbose=1)(delayed(self._tokenize)(sentence, sentence_length, add_sos, add_eos) for sentence in sentences)
#         # return list(zip(*result))  # tokens, masks
#         return result

#     def _tokenize(self, sentence, sentence_length=None, add_sos=False, add_eos=False):
#         # SOS and EOS don't count towards the sentence length
#         # self.mask = []
#         if isinstance(sentence, str):
#             sentence = utils.normalizeString(sentence)
#             sentence = sentence.strip().split()
#             # sentence, mask = self._process_string_list(sentence, sentence_length, add_sos, add_eos)
#             sentence = self._process_string_list(sentence, sentence_length, add_sos, add_eos)
#             return [self.word2idx.get(word, self.reserved_word2idx['<UNK>']) for word in sentence]  # , mask
#         else:
#             sentence = [(self.idx2word[idx] if idx < len(self) else self.reserved_idx2word['<UNK>']) for idx in sentence]
#             # sentence, mask = self._process_string_list(sentence, sentence_length, add_sos, add_eos)
#             sentence = self._process_string_list(sentence, sentence_length, add_sos, add_eos)
#             sentence = ' '.join(sentence)
#             return sentence  #, mask

#     def _process_string_list(self, sentence, length, add_sos, add_eos):
#         if add_sos and sentence[0] != '<SOS>':
#             sentence = ['<SOS>'] + sentence
#         if add_eos and sentence[-1] != '<EOS>':
#             sentence.append('<EOS>')
#         # mask = [1] * len(sentence)
#         if length is not None:
#             sentence = sentence[:length]
#             # mask = mask[:length]
#             padding = length - len(sentence)
#             sentence.extend(['<PAD>'] * padding)
#             # mask.extend([0] * padding)
#         return sentence  # , mask


#     def cleanup(self, sentences):
#         # Normalize
#         return Parallel(n_jobs=-1, verbose=1)(delayed(utils.normalizeString)(sentence) for sentence in sentences)

#     # def __call__(self, sentences):
#     #     return self.tokenize(sentences)
