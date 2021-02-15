"""
Byte pair encoding utilities from GPT-2.

Original source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
Original license: MIT
"""

import json
from functools import lru_cache


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors="replace"):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        try:
            import regex as re

            self.re = re
        except ImportError:
            raise ImportError("Please install regex with: pip install regex")

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = self.re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    ###
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    #단어와 token의 matching 여부를 파악하기 위함
    def encode(self, text):
        orig_text = text
        char2token = []
        token2startchar = []
        token2endchar = []
        bpe_tokens = []
        
        """
        original text에서의 하나의 token이 두 개 이상 bpe token으로 표현되는 경우가 존재.
        이 경우 char2token에서는 해당 token의 모든 char이 첫번째 bpe token의 index를 가리킴
        token2startchar, token2endchar의 경우 모든 bpe token들이 해당 token의 첫번째, 마지막 character index를 동일하게 가리킴
        
        """
        
        char_idx = 0
        for token in self.re.findall(self.pat, text):
            if orig_text.startswith(token):
                orig_text = orig_text[len(token):]
                char_start_idx = char_idx
                char_idx += len(token)
                char_end_idx = char_idx-1
                before_token = token
                token = "".join(self.byte_encoder[b] for b in token.encode("utf-8")) ####token 길이가 달라짐
                for c in before_token:
                    char2token.append(len(bpe_tokens))
                for bpe_token in self.bpe(token).split(" "):
                    token2startchar.append(char_start_idx)
                    token2endchar.append(char_end_idx)                    
                    bpe_tokens.append(self.encoder[bpe_token])
            
        assert len(bpe_tokens)==len(token2startchar) and len(bpe_tokens)==len(token2endchar)
        assert len(text)==len(char2token)
        return bpe_tokens, char2token, token2startchar, token2endchar

    def decode(self, tokens):
        text = "".join([self.decoder.get(token, token) for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            "utf-8", errors=self.errors
        )
        return text


def get_encoder(encoder_json_path, vocab_bpe_path):
    with open(encoder_json_path, "r") as f:
        encoder = json.load(f)
    with open(vocab_bpe_path, "r", encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )
