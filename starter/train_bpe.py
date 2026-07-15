"""Train a byte-level BPE tokenizer on train_corpus.txt ONLY.

Lossless by construction: merges only glue adjacent byte spans together,
so decode() is just concatenation of the underlying bytes -> exact
round trip on ANY input, including bytes/pairs never seen in training
(they simply stay as raw byte tokens, 0-255 fallback preserved).

Why this matters for this corpus: it's mixed English + Hindi. The byte
tokenizer burns 3 bytes per Devanagari codepoint (UTF-8), so the Hindi
half of the corpus eats 3x the context budget of the English half for
the same amount of "meaning". A BPE trained on this corpus will quickly
learn common Devanagari codepoint sequences (and common English words)
as single tokens, roughly equalizing bytes-per-token across scripts and
giving the model a much longer effective context at the same block_size.

Usage: python train_bpe.py --data ../data/train_corpus.txt --vocab_size 768 --out bpe_merges.json
"""
import argparse
import json
import re
import time
import collections

WORD_RE = re.compile(r"\s+|\S+")


def get_chunk_counts(text):
    chunks = WORD_RE.findall(text)
    return collections.Counter(chunks)


def train_bpe(text, vocab_size, verbose=True):
    assert vocab_size >= 256
    n_merges = vocab_size - 256
    counts = get_chunk_counts(text)

    # represent each unique chunk as a tuple of byte ids, keep its count
    words = {}  # tuple(bytes) -> count
    for chunk, cnt in counts.items():
        words[tuple(chunk.encode("utf-8"))] = cnt

    merges = []  # list of ((a,b), new_id) in order learned
    next_id = 256
    t0 = time.time()

    for m in range(n_merges):
        pair_counts = collections.Counter()
        for word, cnt in words.items():
            if len(word) < 2:
                continue
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += cnt

        if not pair_counts:
            break
        best_pair, best_count = pair_counts.most_common(1)[0]
        if best_count < 2:
            break

        merges.append((best_pair, next_id))
        new_words = {}
        a, b = best_pair
        for word, cnt in words.items():
            if a in word and b in word:
                out = []
                i = 0
                L = len(word)
                while i < L:
                    if i < L - 1 and word[i] == a and word[i + 1] == b:
                        out.append(next_id)
                        i += 2
                    else:
                        out.append(word[i])
                        i += 1
                new_words[tuple(out)] = new_words.get(tuple(out), 0) + cnt
            else:
                new_words[word] = new_words.get(word, 0) + cnt
        words = new_words
        next_id += 1

        if verbose and (m + 1) % 64 == 0:
            print(f"  merge {m+1}/{n_merges}  pair={best_pair} count={best_count}  "
                  f"({time.time()-t0:.0f}s)")

    if verbose:
        print(f"trained {len(merges)} merges in {time.time()-t0:.0f}s, "
              f"final vocab={256+len(merges)}")
    return merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab_size", type=int, default=768)
    ap.add_argument("--out", default="bpe_merges.json")
    args = ap.parse_args()

    text = open(args.data, encoding="utf-8").read()
    merges = train_bpe(text, args.vocab_size)

    with open(args.out, "w") as f:
        json.dump({"vocab_size": 256 + len(merges),
                    "merges": [[list(p), nid] for p, nid in merges]}, f)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()