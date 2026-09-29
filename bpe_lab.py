"""
Integrated BPE lab CLI.

Examples (from repo root):
  python bpe_lab.py train-word
  python bpe_lab.py encode-word --text "hello world"
  python bpe_lab.py train-byte
  python bpe_lab.py encode-byte --text "Hello!"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bpe_core import (
    decode_byte_bpe_with_merges,
    decode_word_bpe,
    encode_byte_bpe,
    encode_word_bpe,
    load_byte_artifacts,
    load_word_artifacts,
    read_corpus,
    save_byte_artifacts,
    save_word_artifacts,
    train_byte_bpe,
    train_word_bpe,
)

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data" / "tiny_corpus.txt"
WORD_ART = ROOT / "artifacts" / "word"
BYTE_ART = ROOT / "artifacts" / "byte"


def cmd_train_word(_: argparse.Namespace) -> None:
    text = read_corpus(CORPUS)
    merges, token_to_id = train_word_bpe(text, vocab_size=400)
    save_word_artifacts(WORD_ART, merges, token_to_id)
    print(f"Saved word artifacts to {WORD_ART} ({len(merges)} merges, vocab {len(token_to_id)})")


def cmd_encode_word(args: argparse.Namespace) -> None:
    merges, token_to_id = load_word_artifacts(WORD_ART)
    ids = encode_word_bpe(args.text, merges, token_to_id)
    print(json.dumps(ids))


def cmd_decode_word(args: argparse.Namespace) -> None:
    _, token_to_id = load_word_artifacts(WORD_ART)
    ids = json.loads(args.ids)
    print(decode_word_bpe(ids, token_to_id))


def cmd_train_byte(_: argparse.Namespace) -> None:
    text = read_corpus(CORPUS)
    merges, token_to_id = train_byte_bpe(text, vocab_size=600)
    save_byte_artifacts(BYTE_ART, merges, token_to_id)
    print(f"Saved byte artifacts to {BYTE_ART} ({len(merges)} merges, vocab {len(token_to_id)})")


def cmd_encode_byte(args: argparse.Namespace) -> None:
    merges, token_to_id = load_byte_artifacts(BYTE_ART)
    ids = encode_byte_bpe(args.text, merges, token_to_id)
    print(json.dumps(ids))


def cmd_decode_byte(args: argparse.Namespace) -> None:
    merges, token_to_id = load_byte_artifacts(BYTE_ART)
    ids = json.loads(args.ids)
    print(decode_byte_bpe_with_merges(ids, token_to_id, merges))


def main() -> None:
    parser = argparse.ArgumentParser(description="BPE tokenizer lab")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train-word", help="Train word-level BPE on data/tiny_corpus.txt").set_defaults(
        func=cmd_train_word
    )

    p = sub.add_parser("encode-word", help="Encode text to token IDs (requires train-word)")
    p.add_argument("--text", required=True)
    p.set_defaults(func=cmd_encode_word)

    p = sub.add_parser("decode-word", help="Decode JSON list of IDs to text")
    p.add_argument("--ids", required=True, help='JSON list, e.g. "[12, 45, 9]"')
    p.set_defaults(func=cmd_decode_word)

    sub.add_parser("train-byte", help="Train byte-level BPE").set_defaults(func=cmd_train_byte)

    p = sub.add_parser("encode-byte", help="Byte-level encode")
    p.add_argument("--text", required=True)
    p.set_defaults(func=cmd_encode_byte)

    p = sub.add_parser("decode-byte", help="Byte-level decode")
    p.add_argument("--ids", required=True)
    p.set_defaults(func=cmd_decode_byte)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
