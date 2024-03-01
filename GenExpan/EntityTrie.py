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


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import os
    model_name="llama-7b"
    model_name=os.path.expanduser(f'~/local_transformers/{model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    enames = [
        "New York",
        "Beijing",
        "New England",
        "Shenzhen",
        "Los Angeles",
        "San Francisco",
        "California",
        "Chicago",
        "Houston",
    ]
    # eos_token_ids = tokenizer.encode('A.\n', add_special_tokens=False)[1:2]
    ents = []
    for ename in enames:
        ent = tokenizer.encode(ename, add_special_tokens=False)
        ents.append(ent)
    trie = EntityTrie(ents)

    print("ename -> etokens:")
    for i, (ename, etokens) in enumerate(zip(enames, ents)):
        print(f"{i}:\t{ename} -> {etokens}")
    print()

    print("test is_prefix():")
    print("'' ", trie.is_prefix(tokenizer.encode("", add_special_tokens=False)))
    print("'New' ", trie.is_prefix(tokenizer.encode("New", add_special_tokens=False)))
    print("'Beijing' ", trie.is_prefix(tokenizer.encode("Beijing", add_special_tokens=False)))
    print("'ABC' ", trie.is_prefix(tokenizer.encode("ABC", add_special_tokens=False)))
    print()

    print("test get_next_tokens():")
    print("'' ", trie.get_next_tokens(tokenizer.encode("", add_special_tokens=False)))
    print("'New' ", trie.get_next_tokens(tokenizer.encode("New", add_special_tokens=False)),)
    print("'Beijing' ", trie.get_next_tokens(tokenizer.encode("Beijing", add_special_tokens=False)))
    print("'ABC' ", trie.get_next_tokens(tokenizer.encode("ABC", add_special_tokens=False)))
    print()

    print("test delete():")
    print("delete 'New York'!")
    trie.delete(tokenizer.encode("New York", add_special_tokens=False))
    print("the next tokens of 'New' are", trie.get_next_tokens(tokenizer.encode("New", add_special_tokens=False)))
    print("delete 'New England'!")
    trie.delete(tokenizer.encode("New England", add_special_tokens=False))
    print("the next tokens of 'New' are", trie.get_next_tokens(tokenizer.encode("New", add_special_tokens=False)))
    print("the next tokens of '' are", trie.get_next_tokens(tokenizer.encode("", add_special_tokens=False)))
    print()
