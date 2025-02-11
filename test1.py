import nltk
nltk.download("words")
from nltk.corpus import words

word_list = words.words()

# Save the list to a text file (one word per line)
with open("word_list.txt", "w", encoding="utf-8") as f:
    for word in word_list:
        f.write(word + "\n")
