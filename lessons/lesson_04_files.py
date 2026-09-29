# Lesson 4: Read training text from a file
# Run from repo root:  python lessons\lesson_04_files.py

from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
corpus_path = repo_root / "data" / "tiny_corpus.txt"

text = corpus_path.read_text(encoding="utf-8")
lines = text.strip().splitlines()

print("Corpus path:", corpus_path)
print("Number of lines:", len(lines))
print("First 3 lines:")
for line in lines[:3]:
    print(" ", line)
