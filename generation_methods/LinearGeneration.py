from math import floor
from typing import Any, List
import itertools

import pandas as pd

from software_under_test.SoftwareUnderTest import SoftwareUnderTest
try:
    # Import CreditCheck types if available so we can special-case it
    from software_under_test.CreditCheck import CreditCheck, Applicant
except Exception:
    CreditCheck = None
    Applicant = None

class LinearGeneration:
    def __init__(self, software_under_test: SoftwareUnderTest, test_suite_size: int = 10, starting_num = 0) -> None:
        self.software_under_test: SoftwareUnderTest = software_under_test
        self.test_suite_size: int = test_suite_size
        self.test_suite: List[Any] = []
        self.curr_num = starting_num
        self.product_list = None
    def generate_int(self, low: int, high: int) -> int:
        # If current number is above the allowed high bound, clamp to high.
        if (self.curr_num > high):
            return high
        # If current number is strictly below the allowed low bound, return low
        # (but allow equality so the starting value is produced and then incremented).
        if (self.curr_num < low):
            return low
        old = self.curr_num
        self.curr_num += 1
        return floor(old)
    def generate_float(self, low: float, high: float, increment: float = 0.5) -> float:
        # Similar bounds handling as generate_int: allow equality so the
        # starting value is returned and then curr_num is incremented.
        if (self.curr_num > high):
            return high
        if (self.curr_num < low):
            return low
        old = self.curr_num
        self.curr_num += increment
        return old
    def generate_test_suite(self):
        self.test_suite = [self.software_under_test.generate_test_from_linear(self) for _ in range(self.test_suite_size)]
    def generate_applicant_tuple(self) -> tuple[int, int, int, int, int, int, int, int]:
        if self.product_list is None or len(self.product_list) == 0:
            citizen_range = range(0, 2)
            state_range = range(0, 2)
            age_range = range(1, 101)
            sex_range = range(0, 2)
            region_range = range(0, 6)
            income_range = range(0, 4)
            dependents_range = range(0, 5)
            marital_range = range(0, 2)

            self.product_list = list(itertools.product(
                citizen_range,
                state_range,
                age_range,
                sex_range,
                region_range,
                income_range,
                dependents_range,
                marital_range,
            ))
            # def sort_key(tup):
                # return (-tup[4], tup[2], tup[6])
            # self.product_list.sort(key=sort_key)
            self.curr_num = 0
        old_num = self.curr_num
        self.curr_num += 1
        return self.product_list[int(old_num)]