import os
import json
from typing import List, Optional, Union
from transformers import PreTrainedTokenizer


class BnGraphemizer:
    def __init__(self):
        self._vowel_diacritics = {
            '\u09be', '\u09bf', '\u09c0', '\u09c1', '\u09c2',
            '\u09c3', '\u09c4', '\u09c7', '\u09c8', '\u09cb', '\u09cc'
        }
        self._connector = '\u09cd'
        self._consonant_diacritics = {
            '\u09bc', '\u09d7'
        }
        self._split_chars = {
            '\u0985', '\u0986', '\u0987', '\u0988', '\u0989', '\u098a',
            '\u098b', '\u098f', '\u0990', '\u0993', '\u0994',
            '\u0995', '\u0996', '\u0997', '\u0998', '\u0999',
            '\u099a', '\u099b', '\u099c', '\u099d', '\u099e',
            '\u099f', '\u09a0', '\u09a1', '\u09a2', '\u09a3',
            '\u09a4', '\u09a5', '\u09a6', '\u09a7', '\u09a8',
            '\u09aa', '\u09ab', '\u09ac', '\u09ad', '\u09ae',
            '\u09af', '\u09b0', '\u09b2', '\u09b6', '\u09b7',
            '\u09b8', '\u09b9', '\u09a1\u09bc', '\u09a2\u09bc',
            '\u09af\u09bc', '\u09b0\u09bc', '\u09b2\u09bc',
            '\u09a8\u09bc',
        }

    def parse(self, word: str) -> List[str]:
        graphemes = []
        i = 0
        while i < len(word):
            char = word[i]
            if char == self._connector:
                if graphemes:
                    graphemes[-1] += char
                i += 1
                if i < len(word):
                    graphemes[-1] += word[i]
                    i += 1
                continue
            grapheme = char
            i += 1
            while i < len(word):
                next_char = word[i]
                if next_char in self._vowel_diacritics or next_char in self._consonant_diacritics:
                    grapheme += next_char
                    i += 1
                elif next_char == self._connector:
                    grapheme += next_char
                    i += 1
                else:
                    break
            graphemes.append(grapheme)
        return graphemes


class BanglaBPETokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs
        )
        self._vocab = {}
        self._ids_to_tokens = {}
        self._bpe_ranks = {}
        self._cache = {}

        if vocab_file and os.path.exists(vocab_file):
            self._load_vocab(vocab_file)

    def _load_vocab(self, vocab_file: str):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self._vocab = json.load(f)
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        vocab_file = os.path.join(save_directory, f"{filename_prefix or ''}vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self._vocab, f, ensure_ascii=False)
        return (vocab_file,)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab.get(self.unk_token, 0))

    def _convert_id_to_token(self, index: int) -> str:
        return self._ids_to_tokens.get(index, self.unk_token)

    def get_vocab(self) -> dict:
        return dict(self._vocab)

    def build_vocab_from_texts(self, texts: List[str], vocab_size: int = 5000):
        char_freq = {}
        for text in texts:
            for ch in text:
                char_freq[ch] = char_freq.get(ch, 0) + 1

        sorted_chars = sorted(char_freq.items(), key=lambda x: -x[1])
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        all_tokens = special_tokens + [c for c, _ in sorted_chars[:vocab_size - len(special_tokens)]]
        self._vocab = {t: i for i, t in enumerate(all_tokens)}
        self._ids_to_tokens = {i: t for t, i in self._vocab.items()}


class BanglaGraphemeTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs
        )
        self.graphemizer = BnGraphemizer()
        self._vocab = {}
        self._ids_to_tokens = {}

        if vocab_file and os.path.exists(vocab_file):
            self._load_vocab(vocab_file)

    def _load_vocab(self, vocab_file: str):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self._vocab = json.load(f)
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        vocab_file = os.path.join(save_directory, f"{filename_prefix or ''}vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self._vocab, f, ensure_ascii=False)
        return (vocab_file,)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        for word in text.split():
            graphemes = self.graphemizer.parse(word)
            tokens.extend(graphemes)
            tokens.append(" ")
        if tokens and tokens[-1] == " ":
            tokens.pop()
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab.get(self.unk_token, 0))

    def _convert_id_to_token(self, index: int) -> str:
        return self._ids_to_tokens.get(index, self.unk_token)

    def get_vocab(self) -> dict:
        return dict(self._vocab)

    def build_vocab_from_texts(self, texts: List[str]):
        grapheme_set = set()
        grapheme_set.update(["<pad>", "<unk>", "<s>", "</s>", " "])
        for text in texts:
            for word in text.split():
                graphemes = self.graphemizer.parse(word)
                for g in graphemes:
                    grapheme_set.add(g)

        sorted_graphemes = sorted(grapheme_set)
        self._vocab = {g: i for i, g in enumerate(sorted_graphemes)}
        self._ids_to_tokens = {i: g for g, i in self._vocab.items()}


def create_tokenizer(tokenizer_type: str, texts: Optional[List[str]] = None, vocab_size: int = 5000):
    if tokenizer_type == "bpe":
        tokenizer = BanglaBPETokenizer()
        if texts:
            tokenizer.build_vocab_from_texts(texts, vocab_size)
        return tokenizer
    elif tokenizer_type == "bng":
        tokenizer = BanglaGraphemeTokenizer()
        if texts:
            tokenizer.build_vocab_from_texts(texts)
        return tokenizer
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
