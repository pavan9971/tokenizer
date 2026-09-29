# Lesson 7: Train, save artifacts, encode/decode round-trip
# Run from repo root:  python lessons\lesson_07_encode_decode.py

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from bpe_core import (
    decode_word_bpe,
    encode_word_bpe,
    read_corpus,
    save_word_artifacts,
    train_word_bpe,
)

corpus = read_corpus(repo_root / "data" / "tiny_corpus.txt")
sample = read_corpus(repo_root / "data" / "sample.txt").strip()

merges, token_to_id = train_word_bpe(corpus, vocab_size=400)
artifacts = repo_root / "artifacts" / "word"
save_word_artifacts(artifacts, merges, token_to_id)

ids = encode_word_bpe(sample, merges, token_to_id)
back = decode_word_bpe(ids, token_to_id)

print("Sample text:", sample)
print("Token IDs (first 20):", ids[:20])
print("Decoded:", back)
print("Round-trip OK:", back == sample)
