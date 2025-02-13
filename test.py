from collections import defaultdict, Counter
from nltk.corpus import brown

# Build a bigram model from the Brown corpus.
bigram_model = defaultdict(Counter)

for sentence in brown.sents():
    # Lowercase all words and add a special start token if desired.
    sentence = [word.lower() for word in sentence]
    for i in range(len(sentence) - 1):
        current_word = sentence[i]
        next_word = sentence[i + 1]
        bigram_model[current_word][next_word] += 1

def predict_next_word(last_word, top_n=4):
    """
    Given the last word, return a list of top_n most likely next words.
    """
    suggestions = bigram_model.get(last_word.lower(), {})
    most_common = suggestions.most_common(top_n)
    return [word for word, count in most_common]

# Example:
print("Predictions for 'the':", predict_next_word("the"))
# Might output: Predictions for 'the': ['other', 'same', 'best', 'only'] (depending on corpus frequencies)
