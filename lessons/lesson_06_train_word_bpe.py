# Lesson 6: Train word-level BPE on tiny_corpus.txt (prints merges)
# Run from repo root:  python lessons\lesson_06_train_word_bpe.py

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from bpe_core import read_corpus, train_word_bpe

corpus = read_corpus(repo_root / "data" / "tiny_corpus.txt")
VOCAB_SIZE = 400

merges, token_to_id = train_word_bpe(corpus, VOCAB_SIZE)

print(f"Learned {len(merges)} merges; vocab size = {len(token_to_id)}")
print("First 15 merges:")
for i, pair in enumerate(merges[:15], start=1):
    print(f"  {i:3d}: {pair[0]!r} + {pair[1]!r} -> {pair[0] + pair[1]!r}")
