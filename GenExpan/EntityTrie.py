import torch


"""
    This is a 'EntityTrie' class used to store all entities.
    Each entity is list[int] or Tensor
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_ent_tokens_end = False


class EntityTrie:
    def __init__(self, ents=None):
        self.root = TrieNode()
        if ents is not None:
            for ent in ents:
                self.insert(ent)

    def insert(self, ent):
        node = self.root
        if isinstance(ent, torch.Tensor):
            ent = ent.cpu().numpy()
        for token in ent:
            if token in node.children:
                node = node.children[token]
            else:
                node.children[token] = TrieNode()
                node = node.children[token]
        node.is_ent_tokens_end = True

    def delete(self, ent):
        length = len(ent)
        # delete recursively
        # if return True,it denotes that the parent-node of the node need to delete the node

        def _delete(node, tokens, i) -> bool:
            if i == length:
                # if this node doesn't denote an entity
                if not node.is_ent_tokens_end:
                    return False
                node.is_ent_tokens_end = False
                return len(node.children) == 0
            # if given tokens is not a prefix of any tokens in trie
            token = tokens[i]
            if token not in node.children or not _delete(
                node.children[token], tokens, i + 1
            ):
                return False
            # if this node has no children nodes,
            # then return True and remove this entry in the parent node.
            del node.children[token]
            return len(node.children) == 0
        _delete(self.root, ent, 0)

    def is_prefix(self, ent) -> bool:
        return self.to_node(ent) is not None

    # if 'ent' is not a prefix, it will return 'None'
    def get_next_tokens(self, ent) -> list | None:
        node = self.to_node(ent)
        return list(node.children) if (node is not None) else None

    def to_node(self, tokens) -> TrieNode | None:
        node = self.root
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        for token in tokens:
            if token not in node.children:
                return None
            node = node.children[token]
        return node

