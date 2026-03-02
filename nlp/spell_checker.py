from typing import Dict, List, Set, Tuple
from pathlib import Path

class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end: bool = False
        
class dict_trie:
    pass

class Trie:
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word: str):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

class SpellChecker:
    def __init__(self, language: str):
        self.trie = Trie()
        self.language = language
        self.load_dictionary()
        
    def load_dictionary(self):
        """Loads words from text dictionary."""
        dict_dir = Path(__file__).parent / "dictionaries"
        dict_dir.mkdir(parents=True, exist_ok=True)
        dict_path = dict_dir / f"{self.language}_words.txt"
        
        # Create a dummy dictionary if doesn't exist
        if not dict_path.exists():
            with open(dict_path, 'w', encoding='utf-8') as f:
                if self.language == 'hi':
                    f.write("भारत\nएक\nदेश\nहै\nहिंदी\nभाषा\n")
                elif self.language == 'ta':
                    f.write("தமிழ்\nவணக்கம்\n")
                elif self.language == 'bn':
                    f.write("বাংলা\nনমস্কার\n")
                else:
                    f.write("hello\nworld\n")
                    
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    self.trie.insert(word)

    def _search_recursive(self, node: TrieNode, ch: str, word: str, row: List[int], results: Set[str], current_word: str, max_cost: int):
        columns = len(word) + 1
        current_row = [row[0] + 1]
        
        for c in range(1, columns):
            insert_cost = current_row[c - 1] + 1
            delete_cost = row[c] + 1
            replace_cost = row[c - 1] + (0 if word[c - 1] == ch else 1)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
            
        if current_row[-1] <= max_cost and node.is_end:
            results.add(current_word)
            
        if min(current_row) <= max_cost:
            for char, child in node.children.items():
                self._search_recursive(child, char, word, current_row, results, current_word + char, max_cost)

    def correct(self, word: str, max_edit_distance: int = 1) -> str:
        """Finds the best correction for a word within max_edit_distance using a Trie edit-distance search."""
        current_row = list(range(len(word) + 1))
        results = set()
        
        # If word is perfectly correct, return it
        node = self.trie.root
        is_exact = True
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                is_exact = False
                break
        if is_exact and node.is_end:
            return word

        for char, child in self.trie.root.children.items():
            self._search_recursive(child, char, word, current_row, results, char, max_edit_distance)
            
        if not results:
            return word
            
        return list(results)[0]
