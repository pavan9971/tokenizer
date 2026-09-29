"""
Core BPE logic for the tokenizer lab (word-level and byte-level).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

END_OF_WORD = "</w>"


def word_to_symbols(word: str) -> tuple[str, ...]:
    if not word:
        return ()
    chars = list(word)
    chars[-1] = chars[-1] + END_OF_WORD
    return tuple(chars)


def pretokenize_words(text: str) -> list[str]:
    return text.split()


def build_word_counts(text: str) -> dict[tuple[str, ...], int]:
    counts: Counter[tuple[str, ...]] = Counter()
    for word in pretokenize_words(text):
        counts[word_to_symbols(word)] += 1
    return dict(counts)


def count_pairs(word_counts: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for word, freq in word_counts.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_pair_in_word(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    a, b = pair
    out: list[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def apply_merges_to_word(word: tuple[str, ...], merges: list[tuple[str, str]]) -> tuple[str, ...]:
    if not merges:
        return word
    rank = {pair: i for i, pair in enumerate(merges)}
    symbols = list(word)
    while True:
        best_idx = None
        best_pos = None
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            if pair not in rank:
                continue
            idx = rank[pair]
            if best_idx is None or idx < best_idx:
                best_idx = idx
                best_pos = i
        if best_pos is None:
            break
        pair = (symbols[best_pos], symbols[best_pos + 1])
        symbols = symbols[:best_pos] + [symbols[best_pos] + symbols[best_pos + 1]] + symbols[best_pos + 2 :]
    return tuple(symbols)


def train_word_bpe(text: str, vocab_size: int) -> tuple[list[tuple[str, str]], dict[str, int]]:
    word_counts = build_word_counts(text)
    vocab: set[str] = set()
    for word in word_counts:
        vocab.update(word)

    merges: list[tuple[str, str]] = []
    while len(vocab) < vocab_size:
        pairs = count_pairs(word_counts)
        if not pairs:
            break
        best_pair, best_count = pairs.most_common(1)[0]
        if best_count < 2:
            break
        merges.append(best_pair)
        vocab.add(best_pair[0] + best_pair[1])
        word_counts = {merge_pair_in_word(w, best_pair): c for w, c in word_counts.items()}

    token_to_id = {tok: i for i, tok in enumerate(sorted(vocab))}
    return merges, token_to_id


def encode_word_bpe(text: str, merges: list[tuple[str, str]], token_to_id: dict[str, int]) -> list[int]:
    ids: list[int] = []
    for word in pretokenize_words(text):
        for tok in apply_merges_to_word(word_to_symbols(word), merges):
            ids.append(token_to_id[tok])
    return ids


def decode_word_bpe(ids: list[int], token_to_id: dict[str, int]) -> str:
    id_to_token = {i: t for t, i in token_to_id.items()}
    text = "".join(id_to_token[i] for i in ids).replace(END_OF_WORD, " ")
    return text.strip()


def _merge_int_pair(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    a, b = pair
    out: list[int] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train_byte_bpe(text: str, vocab_size: int) -> tuple[list[tuple[int, int]], dict[int, int]]:
    blob = tuple(text.encode("utf-8"))
    word_counts: dict[tuple[int, ...], int] = {blob: 1}
    vocab: set[int] = set(range(256))
    merges: list[tuple[int, int]] = []

    while len(vocab) < vocab_size:
        pair_counter: Counter[tuple[int, int]] = Counter()
        for word, freq in word_counts.items():
            for i in range(len(word) - 1):
                pair_counter[(word[i], word[i + 1])] += freq
        if not pair_counter:
            break
        best_pair, best_count = pair_counter.most_common(1)[0]
        if best_count < 2:
            break
        new_id = 256 + len(merges)
        merges.append(best_pair)
        vocab.add(new_id)
        word_counts = {_merge_int_pair(w, best_pair, new_id): c for w, c in word_counts.items()}

    token_to_id = {tok: i for i, tok in enumerate(sorted(vocab))}
    return merges, token_to_id


def encode_byte_bpe(text: str, merges: list[tuple[int, int]], token_to_id: dict[int, int]) -> list[int]:
    rank = {pair: i for i, pair in enumerate(merges)}
    merged_id = {pair: 256 + i for i, pair in enumerate(merges)}
    symbols: list[int] = list(text.encode("utf-8"))

    while True:
        best_rank = None
        best_pos = None
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            if pair not in rank:
                continue
            r = rank[pair]
            if best_rank is None or r < best_rank:
                best_rank = r
                best_pos = i
        if best_pos is None:
            break
        pair = (symbols[best_pos], symbols[best_pos + 1])
        symbols = symbols[:best_pos] + [merged_id[pair]] + symbols[best_pos + 2 :]

    return [token_to_id[s] for s in symbols]


def decode_byte_bpe_with_merges(
    ids: list[int], token_to_id: dict[int, int], merges: list[tuple[int, int]]
) -> str:
    id_to_token = {i: t for t, i in token_to_id.items()}

    def expand(token: int) -> list[int]:
        if token < 256:
            return [token]
        idx = token - 256
        if idx < 0 or idx >= len(merges):
            return [token]
        a, b = merges[idx]
        return expand(a) + expand(b)

    byte_vals: list[int] = []
    for idx in ids:
        byte_vals.extend(expand(id_to_token[idx]))
    return bytes(byte_vals).decode("utf-8", errors="replace")


def read_corpus(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def save_word_artifacts(
    path: Path, merges: list[tuple[str, str]], token_to_id: dict[str, int]
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "merges.txt").open("w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")
    with (path / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(token_to_id, f, ensure_ascii=False, indent=2)


def load_word_artifacts(path: Path) -> tuple[list[tuple[str, str]], dict[str, int]]:
    merges: list[tuple[str, str]] = []
    with (path / "merges.txt").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split(" ", 1)
            merges.append((a, b))
    with (path / "vocab.json").open(encoding="utf-8") as f:
        token_to_id = json.load(f)
    return merges, token_to_id


def save_byte_artifacts(
    path: Path, merges: list[tuple[int, int]], token_to_id: dict[int, int]
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "byte_merges.txt").open("w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")
    with (path / "byte_vocab.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in token_to_id.items()}, f, indent=2)


def load_byte_artifacts(path: Path) -> tuple[list[tuple[int, int]], dict[int, int]]:
    merges: list[tuple[int, int]] = []
    with (path / "byte_merges.txt").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split()
            merges.append((int(a), int(b)))
    with (path / "byte_vocab.json").open(encoding="utf-8") as f:
        raw = json.load(f)
    token_to_id = {int(k): v for k, v in raw.items()}
    return merges, token_to_id
