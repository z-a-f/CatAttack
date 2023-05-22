import random

class TranslationDataset:
    def __init__(self,
                 src=None, dst=None,  # Assume that src[idx] and dst[idx] are the same thing
                 src_name=None, dst_name=None,
                 src_max_length=None, dst_max_length=None,
                 src_valid_prefixes=None, dst_valid_prefixes=None,
                 src_tokenizer=None, dst_tokenizer=None,
                 shuffle=True,
                 sentence_length=None):
        self.src_raw = src
        self.dst_raw = dst
        self.dst_name = dst_name or 'dst'
        self.src_name = src_name or 'src'

        # Cleanup the sentences
        # self.src_tokenizer = Lang(name=self.src_name)
        # self.dst_tokenizer = Lang(name=self.src_name)
        self.src_tokenizer = src_tokenizer
        self.dst_tokenizer = dst_tokenizer

        if self.src_raw is None or self.dst_raw is None:
            self.src = None
            self.dst = None
            self.src_tokenized = None
            self.dst_tokenized = None
            return

        assert len(src) == len(dst)

        data = zip(self.src_tokenizer.cleanup(self.src_raw), self.dst_tokenizer.cleanup(self.dst_raw))

        if shuffle:
            data = list(data)
            random.shuffle(data)

        # Filter the sentences by valid prefixes
        if src_valid_prefixes:
            data = filter(lambda pair: pair[0].startswith(src_valid_prefixes), data)
        if dst_valid_prefixes:
            data = filter(lambda pair: pair[1].startswith(dst_valid_prefixes), data)

        # Filter by number of tokens
        if src_max_length is not None:
            data = filter(lambda pair: len(pair[0]) < src_max_length, data)
        if dst_max_length is not None:
            data = filter(lambda pair: len(pair[1]) < dst_max_length, data)

        self.src, self.dst = list(zip(*data))
        if sentence_length is not None:
            # Maximum sentence length = length + 2
            src_max_len = 0
            dst_max_len = 0
            for idx in range(len(self.src)):
                src_max_len = max(src_max_len, len(self.src[idx].split()))
                dst_max_len = max(dst_max_len, len(self.dst[idx].split()))
            src_max_len = min(sentence_length, src_max_len+2)
            dst_max_len = min(sentence_length, dst_max_len+2)
        else:
            src_max_len=None
            dst_max_len=None

        # Tokenizers
        self.src_tokenizer.add(self.src).make_tokens()
        self.dst_tokenizer.add(self.dst).make_tokens()

        # Tokenize the data
        self.src_tokenized, self.src_mask = self.src_tokenizer.tokenize(self.src, sentence_length=src_max_len, add_sos=False, add_eos=True)
        self.dst_tokenized, self.dst_mask = self.dst_tokenizer.tokenize(self.dst, sentence_length=dst_max_len, add_sos=True, add_eos=True)

    def __len__(self):
        assert len(self.src) == len(self.dst)
        return len(self.src)

    def __getitem__(self, idx):
        return ((self.src_tokenized[idx], self.src_mask[idx]),
                (self.dst_tokenized[idx], self.dst_mask[idx]))

    def split(self, ratio):
        left = TranslationDataset()
        right = TranslationDataset()

        end_idx = int(len(self) * ratio)

        for direction in ['src', 'dst']:
            # Split
            for attr in ['', '_raw', '_tokenized', '_mask']:
                attr_name = f'{direction}{attr}'
                setattr(left, attr_name, getattr(self, attr_name)[:end_idx])
                setattr(right, attr_name, getattr(self, attr_name)[end_idx:])
            # Copy
            for attr in ['_name', '_tokenizer']:
                attr_name = f'{direction}{attr}'
                setattr(left, attr_name, getattr(self, attr_name))
                setattr(right, attr_name, getattr(self, attr_name))
        return left, right
