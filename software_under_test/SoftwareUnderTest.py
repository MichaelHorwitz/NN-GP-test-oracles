from typing import List
from typing import Any

import pandas as pd
from gp.Node import Node
from abc import ABC

class SoftwareUnderTest(ABC):
    def __init__(self):
        self.total_branches = 0
    def generate_test_from_tree(self, tree: Node) -> Any:
        ...
    def generate_test_from_random(self, rng: Any) -> Any:
        ...
    def generate_test_from_linear(self, linear: Any) -> Any:
        ...
    def find_coverage(self, test_suite) -> float:
        ...
    def program_input_to_pandas(self, inputs: List[Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
        ...
    def get_num_outputs(self) -> int:
        ...
    def get_tailored_suite(self) -> Any:
        ...