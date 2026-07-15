"""Byte-level BPE tokenizer, trained ONLY on train_corpus.txt (see
train_bpe.py). Falls back gracefully to raw bytes for anything unseen,
so it is lossless on ARBITRARY UTF-8 text: decode(encode(text)) == text
always, because merges only glue adjacent byte spans together and
decode is just concatenation of the underlying bytes.

Interface required by train.py / evaluate.py:
    load() -> tokenizer with .encode(str)->list[int], .decode(list[int])->str,
    .vocab_size. Called with NO arguments; merges file resolved relative to
    this file so grading (cwd = submission folder) still finds it.
"""
import json
import os
import re

WORD_RE = re.compile(r"\s+|\S+")

_MERGES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "bpe_merges.json")


class ByteTokenizer:
    """Fallback: raw UTF-8 bytes, vocab 256. Used if no merges file exists."""
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


class BPETokenizer:
    def __init__(self, merges):
        # merges: list of [[a, b], new_id] in the order they were learned
        self.merge_id = {}      # (a, b) -> new_id
        self.rank = {}          # (a, b) -> priority (lower = merge first)
        self.id_bytes = {i: bytes([i]) for i in range(256)}
        for rank, (pair, new_id) in enumerate(merges):
            a, b = pair
            pair = (a, b)
            self.merge_id[pair] = new_id
            self.rank[pair] = rank
            self.id_bytes[new_id] = self.id_bytes[a] + self.id_bytes[b]
        self.vocab_size = 256 + len(merges)

    def _encode_word(self, word_bytes):
        ids = list(word_bytes)
        while len(ids) > 1:
            pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
            ranked = [(self.rank[p], i) for i, p in enumerate(pairs)
                      if p in self.rank]
            if not ranked:
                break
            _, idx = min(ranked, key=lambda t: t[0])
            pair = pairs[idx]
            new_id = self.merge_id[pair]
            ids = ids[:idx] + [new_id] + ids[idx + 2:]
        return ids

    def encode(self, text):
        out = []
        for chunk in WORD_RE.findall(text):
            out.extend(self._encode_word(tuple(chunk.encode("utf-8"))))
        return out

    def decode(self, ids):
        return b"".join(self.id_bytes[i] for i in ids).decode(
            "utf-8", errors="replace")

    def save(self, path):
        merges = sorted(self.rank.items(), key=lambda kv: kv[1])
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size,
                        "merges": [[list(p), self.merge_id[p]] for p, _ in merges]},
                       f)


def load(path=None):
    """Return the tokenizer used by train.py / evaluate.py."""
    p = path or _MERGES_FILE
    if os.path.exists(p):
        with open(p) as f:
            data = json.load(f)
        return BPETokenizer(data["merges"])
    return ByteTokenizer()