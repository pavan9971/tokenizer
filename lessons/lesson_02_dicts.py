# Lesson 2: Dictionaries count things (used for pair frequencies in BPE)
# Run: python lessons\lesson_02_dicts.py

pairs = [("h", "e"), ("e", "l"), ("l", "l"), ("l", "o"), ("h", "e")]

counts: dict[tuple[str, str], int] = {}
for pair in pairs:
    if pair not in counts:
        counts[pair] = 0
    counts[pair] = counts[pair] + 1

print("Pair counts:", counts)

best_pair = None
best_count = 0
for pair, count in counts.items():
    if count > best_count:
        best_count = count
        best_pair = pair

print("Most common pair:", best_pair, "count =", best_count)
