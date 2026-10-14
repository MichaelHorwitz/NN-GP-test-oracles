from collections import defaultdict
from typing import Any, List, Set

import pandas as pd
from pandas.core.api import DataFrame as DataFrame
from generation_methods.LinearGeneration import LinearGeneration
from generation_methods.RandomSampler import RandomSampler
from gp.Node import Node
from software_under_test.SoftwareUnderTest import SoftwareUnderTest


class Collatz(SoftwareUnderTest):
    def __init__(self):
        super().__init__()
        self.number_branches = 7
        self.__name__ = "Collatz"
    def generate_test_from_tree(self, tree: Node) -> int:
        # Compute the value and ensure it's within reasonable bounds
        value = tree.do_compute()
        # Convert to int and clip to prevent extreme values
        try:
            value = int(value)
        except (ValueError, OverflowError):
            value = 0
        # Clip to reasonable range for Collatz (0 to 10_000_000)
        value = max(0, min(value, 10_000_000))
        return value
    def generate_test_from_random(self, rng: RandomSampler) -> Any:
        return rng.generate_int(0, 10_000)
    def generate_test_from_linear(self, linear: LinearGeneration) -> Any:
        return linear.generate_int(0,10_000)
    def find_coverage(self, test_suite: List[int]) -> float:
        set_of_branches: Set[str] = set()
        for test in test_suite:
            list_of_branches = self.exec_sut_coverage(test)
            for branch in list_of_branches:
                set_of_branches.add(branch)
        return len(set_of_branches) / self.number_branches

    def program_input_to_pandas(self, inputs: List[Any]) -> tuple[DataFrame, DataFrame]:
        outputs = [self.exec_real_output(input) for input in inputs]
        inputs_df = pd.DataFrame({
            'input': inputs
        })
        outputs_df = pd.DataFrame({
            'outputs': outputs
        })
        return (inputs_df,outputs_df)

    def exec_sut_coverage(self, number: int) -> List[str]:
        return_set: List[str] = []
        p = 0
        if number == 0:
            number = 1
        if number <= 0:
            # raise ValueError("Number must be positive.")
            number *= -1
        while number != 1:
            if number % 2 == 0:
                p += 1
                number //= 2
            else:
                p -= 1
                number = 3 * number + 1

        if p < 5:
            return_set.append("RED")
        elif p < 15:
            return_set.append("ORANGE")
        elif p < 30:
            return_set.append("YELLOW")
        elif p < 40:
            return_set.append("GREEN")
        elif p < 50:
            return_set.append("BLUE")
        elif p < 60:
            return_set.append("PINK")
        elif p < 70:
            return_set.append("PURPLE")

        return return_set
    
    def exec_real_output(self, number: int) -> int:
        p = 0
        if number == 0:
            number = 1
        if number <= 0:
            # raise ValueError("Number must be positive.")
            number *= -1
        while number != 1:
            if number % 2 == 0:
                p += 1
                number //= 2
            else:
                p -= 1
                number = 3 * number + 1
        return p
    def get_num_outputs(self) -> int:
        return -1
    def get_tailored_suite(self):
        input_set = [x for x in range(100)]
        output_set = [self.exec_real_output(x) for x in input_set]
        binned_data = defaultdict(list)
        for output in output_set:
            if output >= 60:
                continue
            key = (output // 10) * 10
            binned_data[key].append(output)
        for key in binned_data.keys():
            binned_data[key].sort()
        fair_list = [binned_data[key][0:2] for key in binned_data.keys()]
        fair_list = [item for sublist in fair_list for item in sublist]
        out_in_dict = {}
        for inp, out in zip(input_set, output_set):
            out_in_dict[out] = inp
        fair_inp = [out_in_dict[out] for out in fair_list]
        return fair_inp, fair_list