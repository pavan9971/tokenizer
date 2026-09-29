# Where you learn (BPE + minimal Python)

This repo **is** the course. Read files in this order.

## 1. Setup

- Install [Python 3.11+](https://www.python.org/downloads/) (check **Add to PATH**).
- Clone [github.com/pavan9971/tokenizer](https://github.com/pavan9971/tokenizer).
- Open the folder in Cursor.

## 2. Python primer (lessons 0–4)

| Lesson | Command | Idea |
|--------|---------|------|
| 0 | `python lessons\lesson_00_hello.py` | Run scripts, `print` |
| 1 | `python lessons\lesson_01_strings_lists.py` | Strings, lists, `split` |
| 2 | `python lessons\lesson_02_dicts.py` | Count with a dict |
| 3 | `python lessons\lesson_03_loops_functions.py` | `for`, `def` |
| 4 | `python lessons\lesson_04_files.py` | Read `data/tiny_corpus.txt` |

**Try this:** change one string in a lesson and run again.

## 3. BPE (lessons 5–8)

| Lesson | Command | Idea |
|--------|---------|------|
| 5 | `python lessons\lesson_05_count_pairs.py` | Adjacent pairs in one word |
| 6 | `python lessons\lesson_06_train_word_bpe.py` | Learn merges from corpus |
| 7 | `python lessons\lesson_07_encode_decode.py` | Save `artifacts/word/`, round-trip |
| 8 | `python lessons\lesson_08_byte_bpe.py` | Bytes 0–255, emoji-friendly |

## 4. What BPE means (short)

1. Start with small pieces (letters + word end marker, or raw bytes).
2. Find the most frequent **pair** of neighbors.
3. Merge that pair into one new token; repeat until vocab is big enough.
4. When encoding new text, apply merges in **the same order** they were learned.

Word-level uses `</w>` on the last character of each word so spaces come back on decode.

## 5. Full CLI (`bpe_lab.py`)

```powershell
python bpe_lab.py train-word
python bpe_lab.py encode-word --text "hello world"
python bpe_lab.py train-byte
python bpe_lab.py encode-byte --text "Hello!"
```

Generated files go under `artifacts/` (gitignored).

## 6. Data

- **Train:** `data/tiny_corpus.txt` (included, ~50 lines).
- **Test:** `data/sample.txt`.
- Replace corpus text with your own notes anytime; re-run train.

## 7. After you finish

Compare with [karpathy/minbpe](https://github.com/karpathy/minbpe) — same ideas, shorter code.
