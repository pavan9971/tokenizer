# Tokenizer lab (BPE from scratch)

Learn **Byte Pair Encoding** and just enough **Python** to build a working tokenizer. No ML frameworks required.

**Repo:** [github.com/pavan9971/tokenizer](https://github.com/pavan9971/tokenizer)

## Quick start

```powershell
cd path\to\tokenizer
python --version
python lessons\lesson_00_hello.py
```

Read the learning map: [docs/LEARN.md](docs/LEARN.md)

## Lesson order

| Step | File |
|------|------|
| Python basics | `lessons/lesson_00_hello.py` … `lesson_04_files.py` |
| BPE step by step | `lesson_05_count_pairs.py` … `lesson_08_byte_bpe.py` |
| All-in-one CLI | `bpe_lab.py` |

## Train and use (after lesson 7)

```powershell
python bpe_lab.py train-word
python bpe_lab.py encode-word --text "tokenizers split text"
```

Byte-level:

```powershell
python bpe_lab.py train-byte
python bpe_lab.py encode-byte --text "Hello!"
```

## Data

Training text lives in `data/tiny_corpus.txt`. You can edit it and re-train. See [docs/LEARN.md](docs/LEARN.md#6-data).

## Layout

```
tokenizer/
  bpe_core.py      # core algorithms (used by lessons + bpe_lab.py)
  bpe_lab.py       # command-line train / encode / decode
  lessons/         # teaching scripts (start here)
  data/            # tiny corpus + sample sentence
  docs/LEARN.md    # where and how to learn
  artifacts/       # generated merges/vocab (not committed)
```

## License

MIT (add a LICENSE file if you publish publicly).
