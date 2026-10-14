from dataclasses import dataclass
from typing import List, Set

import pandas as pd
from keras.utils import to_categorical
from generation_methods.LinearGeneration import LinearGeneration
from generation_methods.RandomSampler import RandomSampler
from gp.Node import Node
from software_under_test.SoftwareUnderTest import SoftwareUnderTest
from enum import Enum, auto
import random
import math

@dataclass
class PossibleTriangle:
    a: float = 0
    b: float = 0     
    c: float = 0

class TriangleType(Enum):
    INVALID = auto()
    EQUILATERAL = auto()
    ISOSCELES = auto()
    SCALENE = auto()
        
class TriangleClassifier(SoftwareUnderTest):
    def __init__(self):
        super().__init__()
        self.number_branches = 6
        self.__name__ = "Triangle"
    def generate_test_from_tree(self, tree: Node) -> PossibleTriangle:
        # Compute values and ensure they're within reasonable bounds
        def _safe_compute():
            try:
                val = tree.do_compute()
                val = float(val)
                # Clip to reasonable range for triangle sides (1 to 10_000)
                return max(1.0, min(val, 10_000.0))
            except (ValueError, OverflowError):
                return 1.0
        
        return PossibleTriangle(
            a=_safe_compute(),
            b=_safe_compute(),
            c=_safe_compute()
        )
    def generate_test_from_random(self, rng: RandomSampler) -> PossibleTriangle:
        a = rng.generate_int(1, 10000)
        b = rng.generate_int(1, 10000)
        c = rng.generate_int(1, 10000)
        return PossibleTriangle(a=a, b=b, c=c)

    def generate_test_from_linear(self, linear: LinearGeneration) -> PossibleTriangle:
        a = linear.generate_int(1, 10000)
        b = linear.generate_int(1, 10000)
        c = linear.generate_int(1, 10000)
        return PossibleTriangle(a=a, b=b, c=c)
    
    def find_coverage(self, test_suite) -> float:
        set_of_branches: Set[str] = set()
        for test in test_suite:
            list_of_branches = self.exec_sut_coverage(test)
            for branch in list_of_branches:
                set_of_branches.add(branch)
        return len(set_of_branches) / self.number_branches
    
    def program_input_to_pandas(self, inputs: list[PossibleTriangle]) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame(inputs)
        X = df.astype('float32')
        y = [self.exec_real_output(tri).value - 1 for tri in inputs]
        y = to_categorical(y, num_classes=4).astype('float32')
        y = pd.DataFrame(y)

        return X, y
    
    def exec_sut_coverage(self, trig: PossibleTriangle) -> List[str]:
        return_set: List[str] = []
        if trig.a <= 0 or trig.b <= 0 or trig.c <= 0:
            return_set.append("RED")
            return return_set
        if trig.a + trig.b <= trig.c or trig.a + trig.c <= trig.b or trig.b + trig.c <= trig.a:
            return_set.append("BLUE")
            return return_set

        return_set.append("PINK")
        if trig.a == trig.b == trig.c:
            return_set.append("GREEN")
        elif trig.a == trig.b or trig.b == trig.c or trig.a == trig.c:
            return_set.append("ORANGE")
        else:
            return_set.append("YELLOW")
        return return_set

    def exec_real_output(self, trig: PossibleTriangle) -> TriangleType:
        if trig.a <= 0 or trig.b <= 0 or trig.c <= 0:
            return TriangleType.INVALID
        if trig.a + trig.b <= trig.c or trig.a + trig.c <= trig.b or trig.b + trig.c <= trig.a:
            return TriangleType.INVALID
        if trig.a == trig.b == trig.c:
            return TriangleType.EQUILATERAL
        elif trig.a == trig.b or trig.b == trig.c or trig.a == trig.c:
            return TriangleType.ISOSCELES
        else:
            return TriangleType.SCALENE
    def get_num_outputs(self) -> int:
        return 4
    def get_tailored_suite(self): # type: ignore
        UPPER_LIMIT = 10
        NUMS_PER_TRIG = 20
        rand_nums = [random.randint(1,UPPER_LIMIT) for _ in range(NUMS_PER_TRIG)]
        equilateral = [(x,x,x) for x in rand_nums]
        rand_sides = [(random.randint(1,UPPER_LIMIT), random.randint(1,UPPER_LIMIT)) for _ in range(NUMS_PER_TRIG)]
        scalene = [(x,y, find_third_side(x,y)) for (x,y) in rand_sides]
        rand_sides = [random.randint(1,UPPER_LIMIT) for _ in range(NUMS_PER_TRIG) for _ in range(NUMS_PER_TRIG)]
        isoc = [(x,x,find_third_side(x,x)) for x in rand_sides]
        rand_sides = [(random.randint(1,UPPER_LIMIT), random.randint(1,UPPER_LIMIT), random.randint(1,UPPER_LIMIT)) for _ in range(NUMS_PER_TRIG)]
        invalid_pos = rand_sides.copy()
        rand_sides = [(random.randint(1,UPPER_LIMIT), random.randint(1,UPPER_LIMIT), random.randint(1,UPPER_LIMIT)) for _ in range(NUMS_PER_TRIG)]
        invalid_neg = [(x, -y, z) for (x,y,z) in rand_sides]
        test_suite = equilateral + scalene + isoc + invalid_pos + invalid_neg
        test_trigs = [PossibleTriangle(a,b,c) for (a,b,c) in test_suite]
        test_features, test_labels = self.program_input_to_pandas(test_trigs)
        return test_features, test_labels

def find_third_side(side_a: float, side_b: float) -> float:
    angle = random.uniform(0, math.pi)
    side_c_squared = (side_a**2) + (side_b**2) - (2 * side_a * side_b * math.cos(angle))
    side_c = math.sqrt(side_c_squared)

    return side_c
