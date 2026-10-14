from enum import Enum
import itertools
from typing import Any, List, Optional
import random

class NodeExceptionType(Enum):
    INCORRECT_NUM_OPERANDS = 0
class NodeException(Exception):
    exception_type: NodeExceptionType
    def __init__(self, type: NodeExceptionType):
        self.exception_type = type


# Hard ceiling for recursive evaluation depth — trees deeper than this
# short-circuit to 0 so a single bloated individual cannot hang the run.
MAX_EVAL_DEPTH = 64
# Clamp magnitude of intermediate values to avoid huge-number arithmetic.
VALUE_CLAMP = 1e15


class Node:
    _id_gen = itertools.count()
    max_children: int
    data: Any
    _eval_depth: int = 0
    _last_output: Any = None
    def __init__(self, *children: 'Node'):
        self.id = next(Node._id_gen)
        if len(children) > self.max_children:
            raise NodeException(NodeExceptionType.INCORRECT_NUM_OPERANDS)
        self.children: List['Node'] = list(children)


    def do_compute(self) -> any: # type: ignore
        Node._eval_depth += 1
        if Node._eval_depth > MAX_EVAL_DEPTH:
            Node._eval_depth -= 1
            return 0
        result = None
        try:
            result = self._compute()
            # Clamp numeric results to prevent huge-number blowup
            if isinstance(result, (int, float)) and not isinstance(result, bool):
                if abs(result) > VALUE_CLAMP:
                    result = VALUE_CLAMP if result > 0 else -VALUE_CLAMP
            return result
        finally:
            Node._eval_depth -= 1
            if Node._eval_depth == 0:
                Node._last_output = result

    @classmethod
    def get_last_output(cls) -> Any:
        return cls._last_output
    
    def _compute(self):
        raise NotImplementedError()

    def add_child(self, new_node: 'Node') -> Optional['Node']:
        if len(self.children) <= self.max_children:
            self.children.append(new_node)
            return new_node
        return None

    def remove_child(self, id: int) -> Optional['Node']:
        for i in range(len(self.children)):
            if self.children[i].id == id:
                return_node = self.children[i]
                self.children.pop(i)
                return return_node
        return None
class OperandNode(Node):
    max_children = 0
    def __init__(self, data):
        super().__init__()
        self.data = data
    def _compute(self):
        return self.data

class AddNode(Node):
    max_children = 2
    def _compute(self):
        if len(self.children) != 2:
            raise NodeException(NodeExceptionType.INCORRECT_NUM_OPERANDS)
        first_val = self.children[0].do_compute()
        second_val = self.children[1].do_compute()
        return first_val + second_val

class DivisionNode(Node):
    max_children = 2
    def _compute(self):
        if len(self.children) != 2:
            raise NodeException(NodeExceptionType.INCORRECT_NUM_OPERANDS)
        numerator = self.children[0].do_compute()
        denominator = self.children[1].do_compute()
        if denominator == 0:
            return 0
        return numerator / denominator  # may raise ZeroDivisionError naturally

class MultiplicationNode(Node):
    max_children = 2
    def _compute(self):
        if len(self.children) != 2:
            raise NodeException(NodeExceptionType.INCORRECT_NUM_OPERANDS)
        return self.children[0].do_compute() * self.children[1].do_compute()

class SubtractionNode(Node):
    max_children = 2
    def _compute(self):
        if len(self.children) != 2:
            raise NodeException(NodeExceptionType.INCORRECT_NUM_OPERANDS)
        return self.children[0].do_compute() - self.children[1].do_compute()

class GreaterThanNode(Node):
    max_children = 2
    def _compute(self):
        if len(self.children) != 2:
            raise NodeException(NodeExceptionType.INCORRECT_NUM_OPERANDS)
        return self.children[0].do_compute() > self.children[1].do_compute()

class LessThanNode(Node):
    max_children = 2
    def _compute(self):
        if len(self.children) != 2:
            raise NodeException(NodeExceptionType.INCORRECT_NUM_OPERANDS)
        return self.children[0].do_compute() < self.children[1].do_compute()

class NotEqualNode(Node):
    max_children = 2
    def _compute(self):
        if len(self.children) != 2:
            raise NodeException(NodeExceptionType.INCORRECT_NUM_OPERANDS)
        return self.children[0].do_compute() != self.children[1].do_compute()

class IfNode(Node):
    # children: [condition, then_branch, else_branch]
    max_children = 3
    def _compute(self):
        if len(self.children) != 3:
            raise NodeException(NodeExceptionType.INCORRECT_NUM_OPERANDS)
        cond = self.children[0].do_compute()
        if cond:
            return self.children[1].do_compute()
        else:
            return self.children[2].do_compute()
        
class RandomNode(Node):
    max_children = 0
    def __init__(self):
        super().__init__()
        self.data = (random.random() - 0.5) * 100
    def _compute(self):
        return (random.random() - 0.5) * 100


class LastOutput(Node):
    max_children = 0
    def _compute(self):
        val = Node.get_last_output()
        return 0 if val is None else val