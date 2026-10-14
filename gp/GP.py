from typing import List
import random
import copy
import logging

from gp.Node import Node
from gp.Tree import Tree
from gp.tree_operations import generate_random_tree, subtree_crossover, subtree_mutation, tree_depth
from software_under_test.SoftwareUnderTest import SoftwareUnderTest

from typing import Callable, Any, Optional

class GeneticProgram:
    def __init__(
        self,
        primative_list: list[type[Node]],
        terminal_list: list[Callable[[], Node]],
        software_under_test: SoftwareUnderTest,
        max_depth: int = 3,
        pop_size: int = 100,
        tournament_select_k: int = 3,
        test_suite_size: int = 10,
        generations: int = 10,
        cx_rate: float = 0.8,
        mut_rate: float = 0.2
    ):
        self.primative_list: list[type[Node]] = primative_list
        self.terminal_list: list[Callable[[], Node]] = terminal_list
        self.software_under_test: SoftwareUnderTest = software_under_test
        self.max_depth: int = max_depth
        self.pop_size: int = pop_size
        self.tree_population: List[Tree] = []  # List of Tree objects
        self.best_test_suite: list[Any] = []
        self.test_suite: list[Any] = []
        self.best_fitness: float = 1000000.0
        self.best_tree: Optional[Node] = None
        self.tournament_select_k: int = tournament_select_k
        self.test_suite_size: int = test_suite_size
        self.generations: int = generations
        self.cx_rate: float = cx_rate
        self.mut_rate: float = mut_rate
        self.logger = logging.getLogger(f"{__name__}.GeneticProgram")

    def init_population(self) -> None:
        """Create initial population"""
        self.logger.debug(f"Generating random population of size: {self.pop_size}")
        # Ensure we pass the configured primitive and terminal lists so the
        # generated trees respect the caller's configuration (for example,
        # integer-only terminals and a reduced primitive set).
        self.tree_population = [Tree(generate_random_tree(self.max_depth, primatives=self.primative_list, terminals=self.terminal_list), 1.0) for _ in range(self.pop_size)]
        i = 0
        for tree in self.tree_population:
            self.logger.debug(f"Evaluating fitness for tree {i}")
            tree.fitness = self.fitness(tree.root_node)
            i += 1


    def tournament_select(self) -> Tree:
        """Select one individual by k‐way tournament."""
        competitors: List[Tree] = random.sample(self.tree_population, self.tournament_select_k)
        winner: Tree = min(competitors, key=lambda cf: cf.fitness)
        return winner
    
    def fitness(self, tree: Node) -> float:
        """Fitness for this GP is inverse coverage"""
        test_suite: list[Any] = [self.software_under_test.generate_test_from_tree(tree) for _ in range(self.test_suite_size)]
        fitness: float = 1 - self.software_under_test.find_coverage(test_suite)
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_test_suite = copy.deepcopy(test_suite)
            self.best_tree = copy.deepcopy(tree)
        return fitness

    def train(self) -> None:
        self.init_population()
        self.logger.info(f"Population Initialised: {self.tree_population[:5]}")
        pop: List[Tree] = self.tree_population
        for gen in range(self.generations):
            if gen % 10 == 0:
                self.logger.info(f"Training Generation: {gen}")
            else: 
                self.logger.debug(f"Training Generation: {gen}")
            new_pop: List[Tree] = []
            while len(new_pop) < self.pop_size:
                p1: Tree = self.tournament_select()
                if random.random() < self.cx_rate:
                    p2: Tree = self.tournament_select()
                    child: Node = subtree_crossover(p1.root_node, p2.root_node)
                else:
                    child: Node = p1.root_node
                if random.random() < self.mut_rate:
                    # Pass the configured terminals/primatives to ensure mutated
                    # subtrees are generated with the same constraints.
                    child = subtree_mutation(child, self.max_depth, terminals=self.terminal_list, primatives=self.primative_list)
                # Bloat control: if the offspring exceeds twice the configured
                # max_depth, discard it and keep one of the parents instead.
                if tree_depth(child) > self.max_depth * 2:
                    child = copy.deepcopy(p1.root_node)
                new_pop.append(Tree(child, self.fitness(child)))
            pop = new_pop
            best_tree: Tree = min(pop, key=lambda i: i.fitness)
            self.logger.debug(f"Gen {gen}: Best fitness = {best_tree.fitness:.4f}")
        self.test_suite = self.best_test_suite
        
    def generate_test_suite(self):
        self.train()
