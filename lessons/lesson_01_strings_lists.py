# Lesson 1: Strings and lists (words become lists of characters)
# Run: python lessons\lesson_01_strings_lists.py

word = "hello"
print("Word:", word)
print("First character:", word[0])

chars = list(word)
print("As a list:", chars)

sentence = "byte pair encoding"
words = sentence.split()
print("Words in sentence:", words)

for w in words:
    print("  letters of", w, "->", list(w))
