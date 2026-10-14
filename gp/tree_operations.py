import copy
import random
from gp.Node import AddNode, DivisionNode, GreaterThanNode, IfNode, LessThanNode, MultiplicationNode, Node, NotEqualNode, OperandNode, RandomNode, SubtractionNode

PRIMITIVES = [AddNode, SubtractionNode, MultiplicationNode, DivisionNode,
              GreaterThanNode, LessThanNode, NotEqualNode, IfNode]
TERMINALS = [ lambda: OperandNode(random.uniform(-100, 100)),  lambda: RandomNode()] 

def generate_random_tree(max_depth, curr_depth=0, terminals=TERMINALS, primatives=PRIMITIVES):
    """Recursively build a random tree up to max_depth."""
    if curr_depth >= max_depth or (curr_depth > 0 and random.random() < 0.3):
        # Return a terminal node
        return random.choice(terminals)()
    else:
        # Choose a random primitive and arity
        NodeClass = random.choice(primatives)
        node = NodeClass()
        for _ in range(NodeClass.max_children):
            # Propagate the same terminals/primatives when recursing so callers'
            # configurations (e.g. integer-only terminals) are respected.
            child = generate_random_tree(max_depth, curr_depth+1, terminals=terminals, primatives=primatives)
            node.add_child(child)
        return node

def get_all_nodes(node):
    """Return flat list of all nodes in the tree (preorder)."""
    nodes = [node]
    for c in node.children:
        nodes.extend(get_all_nodes(c))
    return nodes

def tree_depth(node) -> int:
    """Return the depth of the tree rooted at *node*."""
    if not node.children:
        return 0
    return 1 + max(tree_depth(c) for c in node.children)

def contains_subtree(node: Node, target_id: int) -> bool:
    """
    Return True if there is any node in the subtree of `node`
    whose id equals `target_id`.
    """
    if node.id == target_id:
        return True
    for child in node.children:
        if contains_subtree(child, target_id):
            return True
    return False

def replace_subtree(root: Node, target_id: int, new_subtree: Node) -> Node:
    """
    Replace the subtree rooted at the node with id == target_id
    by `new_subtree`, returning the (possibly updated) root.
    """
    if root.id == target_id:
        return new_subtree

    for idx, child in enumerate(root.children):
        if contains_subtree(child, target_id):
            # Direct replacement in parent's children list
            root.children[idx] = new_subtree
            return root

    return root

def subtree_crossover(parent1: Node, parent2: Node) -> Node:
    p1 = copy.deepcopy(parent1)
    p2 = copy.deepcopy(parent2)
    n1 = random.choice(get_all_nodes(p1))
    n2 = random.choice(get_all_nodes(p2))
    replace_subtree(p1, n1.id, n2)
    return p1

def subtree_mutation(tree, max_depth, terminals=TERMINALS, primatives=PRIMITIVES):
    """Replace a random subtree with a newly-generated subtree.

    Allow callers to pass the terminals/primatives to ensure mutations use the
    same configuration as the original generation (avoids falling back to the
    module defaults which may introduce floats or unwanted primitives).
    """
    t = copy.deepcopy(tree)
    target = random.choice(get_all_nodes(t))
    new_sub = generate_random_tree(max_depth, terminals=terminals, primatives=primatives)
    replace_subtree(t, target.id, new_sub)
    return t