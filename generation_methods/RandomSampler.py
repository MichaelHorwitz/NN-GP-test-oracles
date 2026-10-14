from math import floor
from random import Random
from typing import Any, List

from software_under_test.SoftwareUnderTest import SoftwareUnderTest

class RandomSampler:
    def __init__(self, software_under_test: SoftwareUnderTest, test_suite_size: int = 10):
        self.random_generator = Random()
        self.software_under_test: SoftwareUnderTest = software_under_test
        self.test_suite_size: int = test_suite_size
        self.test_suite: List[Any] = []
    def generate_int(self, low: int, high: int) -> int:
        return floor(self.random_generator.uniform(low, high))
    def generate_float(self, low: float, high: float) -> float:
        return self.random_generator.uniform(low, high)
    def generate_test_suite(self):
        self.test_suite = [self.software_under_test.generate_test_from_random(self) for _ in range(self.test_suite_size)]