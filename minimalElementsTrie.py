'''
using a trie to store the sets within the upset
'''

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_set = False  # This flags the completion of a set

class Trie:
    def __init__(self):
        self.root = TrieNode()

