from typing import List, Dict, Optional, Set

class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_word: bool = False
        self.word: Optional[str] = None

class LexiconTrie:
    """
    A Trie structure to store a lexicon of valid words.
    Used for phoneme-to-word decoding.
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        """Inserts a word into the Trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.word = word

    def search(self, prefix: str) -> bool:
        """Checks if a prefix exists in the Trie."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def get_completions(self, prefix: str) -> List[str]:
        """Returns all words in the Trie that start with the given prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        completions = []
        self._dfs(node, completions)
        return completions

    def _dfs(self, node: TrieNode, results: List[str]):
        if node.is_word:
            results.append(node.word)
        for child in node.children.values():
            self._dfs(child, results)

def load_lexicon_from_file(file_path: str) -> LexiconTrie:
    """Loads a lexicon from a vocab.txt file."""
    trie = LexiconTrie()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                # Skip special tokens and subwords
                if word.startswith('[') or word.startswith('##') or len(word) < 2:
                    continue
                # Only include word-like strings (Vietnamese specific filtering can be added)
                trie.insert(word)
    except Exception as e:
        print(f"❌ Failed to load lexicon: {e}")
    return trie
