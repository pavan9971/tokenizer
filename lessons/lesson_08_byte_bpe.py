# Lesson 8: Byte-level BPE (UTF-8 bytes, works with emoji)
# Run from repo root:  python lessons\lesson_08_byte_bpe.py

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from bpe_core import (
    decode_byte_bpe_with_merges,
    encode_byte_bpe,
    read_corpus,
    save_byte_artifacts,
    train_byte_bpe,
)

corpus = read_corpus(repo_root / "data" / "tiny_corpus.txt")
sample = "Hello byte BPE!"

merges, token_to_id = train_byte_bpe(corpus, vocab_size=600)
save_byte_artifacts(repo_root / "artifacts" / "byte", merges, token_to_id)

for text in [sample, "Emoji: \U0001f600"]:
    ids = encode_byte_bpe(text, merges, token_to_id)
    back = decode_byte_bpe_with_merges(ids, token_to_id, merges)
    label = text.encode("unicode_escape").decode("ascii")
    print("Text:", label)
    print("  IDs count:", len(ids))
    print("  Round-trip OK:", back == text)
