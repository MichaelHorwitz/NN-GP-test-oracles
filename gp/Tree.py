from gp.Node import Node


class Tree:
    def __init__(self, root_node: Node, fitness: float):
        self.root_node: Node = root_node
        self.fitness: float = fitness
