# Lesson 3: Loops and functions (training loop building blocks)
# Run: python lessons\lesson_03_loops_functions.py


def word_to_symbols(word: str) -> tuple[str, ...]:
    """Characters with </w> on the last letter (word boundary marker)."""
    end_marker = "</w>"
    if not word:
        return ()
    chars = list(word)
    chars[-1] = chars[-1] + end_marker
    return tuple(chars)


def count_adjacent_pairs(symbols: tuple[str, ...]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for i in range(len(symbols) - 1):
        pair = (symbols[i], symbols[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


word = "hello"
symbols = word_to_symbols(word)
print("Symbols:", symbols)
print("Pairs:", count_adjacent_pairs(symbols))
