# Lesson 5: Count character pairs in one word (BPE building block)
# Run: python lessons\lesson_05_count_pairs.py

from collections import Counter


def word_to_symbols(word: str) -> tuple[str, ...]:
    end_marker = "</w>"
    chars = list(word)
    chars[-1] = chars[-1] + end_marker
    return tuple(chars)


def count_pairs(symbols: tuple[str, ...]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for i in range(len(symbols) - 1):
        pairs[(symbols[i], symbols[i + 1])] += 1
    return pairs


word = "hello"
symbols = word_to_symbols(word)
pairs = count_pairs(symbols)

print("Word symbols:", symbols)
print("Pair counts:", dict(pairs))
print("Most common pair:", pairs.most_common(1)[0])
