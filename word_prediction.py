from heapq import heappush, heappop

# Load the list from the text file
with open("word_list.txt", "r", encoding="utf-8") as f:
    word_list = [line.strip() for line in f]


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

# Initialize and populate Trie
trie = Trie()
for word in word_list:
    trie.insert(word.lower())

typ = ""
while(typ != "x"):
    typ = input()
    prefix = typ.strip().split()[-1]
    suggestions = trie.search(prefix)
    print(f"Suggestions for '{prefix}':", suggestions)
