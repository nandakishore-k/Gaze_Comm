import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit, QLabel, QVBoxLayout, QWidget
import nltk
from heapq import heappush, heappop

nltk.download("words")
from nltk.corpus import words

# Load English words into a set
word_list = set(words.words())


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.word = None


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.word = word

    def search(self, prefix, max_results=4):
        """Find top word completions for a given prefix."""
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # No words with this prefix

        return self._collect_words(node, max_results)

    def _collect_words(self, node, max_results):
        """Use a priority queue to efficiently fetch top words."""
        heap, results = [], []

        def dfs(n):
            if n.is_end:
                heappush(heap, n.word)
            for child in n.children.values():
                dfs(child)
                if len(heap) > max_results:
                    heappop(heap)

        dfs(node)
        while heap:
            results.append(heappop(heap))

        return results[::-1]  # Return most relevant words first


# Initialize Trie with words
trie = Trie()
for word in word_list:
    trie.insert(word.lower())


# ---------------------- PyQt UI ----------------------
class OptiTypeApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("OptiType Word Prediction")
        self.setGeometry(100, 100, 400, 200)

        # Create UI elements
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Type here...")

        self.prediction_label = QLabel("Predictions: ", self)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.input_field)
        layout.addWidget(self.prediction_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Connect text input change
        self.input_field.textChanged.connect(self.update_predictions)

    def update_predictions(self):
        """Fetch and update predictions dynamically."""
        prefix = self.input_field.text().strip().lower()
        if prefix:
            suggestions = trie.search(prefix)
            self.prediction_label.setText(f"Predictions: {', '.join(suggestions)}")
        else:
            self.prediction_label.setText("Predictions: ")


# Run Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OptiTypeApp()
    window.show()
    sys.exit(app.exec_())
