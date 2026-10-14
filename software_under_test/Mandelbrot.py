from typing import Any, List, Tuple
import pandas as pd
from software_under_test.SoftwareUnderTest import SoftwareUnderTest
from generation_methods.RandomSampler import RandomSampler
from generation_methods.LinearGeneration import LinearGeneration

class Mandelbrot(SoftwareUnderTest):
    """SoftwareUnderTest wrapper for a Mandelbrot pixel->color function.

    The real output function expects an input tuple describing a pixel and
    canvas: (px, py, width, height, max_iteration=1000, x_min=-2.0, x_max=0.47, y_min=-1.12, y_max=1.12)
    It returns an (R, G, B) tuple with values in 0-255.
    """
    def __init__(self):
        super().__init__()
        self.number_branches = 0
        self.__name__ = "Mandelbrot"

    def generate_test_from_tree(self, tree: Any) -> tuple:
        # Expect tree.do_compute to produce numeric values; produce a pixel tuple
        px = int(tree.do_compute())
        py = int(tree.do_compute())
        width = int(tree.do_compute())
        height = int(tree.do_compute())
        return (px, py, width, height)

    def generate_test_from_random(self, rng: RandomSampler) -> tuple:
        width = rng.generate_int(100, 800)
        height = rng.generate_int(100, 800)
        px = rng.generate_int(0, width - 1)
        py = rng.generate_int(0, height - 1)
        return (px, py, width, height)

    def generate_test_from_linear(self, linear: LinearGeneration) -> tuple:
        width = int(linear.generate_int(100, 800))
        height = int(linear.generate_int(100, 800))
        px = int(linear.generate_int(0, width - 1))
        py = int(linear.generate_int(0, height - 1))
        return (px, py, width, height)

    def find_coverage(self, test_suite: List[tuple]) -> float:
        # Not applicable for Mandelbrot; return 0.0
        return 0.0

    def program_input_to_pandas(self, inputs: List[tuple]) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Convert tuple inputs into a dataframe of features and RGB labels
        df = pd.DataFrame(inputs, columns=["px", "py", "width", "height"])
        X = df.astype('int32')
        y = [self.exec_real_output(inp) for inp in inputs]
        y_df = pd.DataFrame(y, columns=["r", "g", "b"]).astype('int32')
        return X, y_df

    def exec_sut_coverage(self, inp: tuple) -> List[str]:
        return []

    def exec_real_output(self, inp: tuple) -> Tuple[int, int, int]:
        """Compute the Mandelbrot color for a single pixel input.

        inp: (px, py, width, height, max_iteration=1000, x_min=-2.0, x_max=0.47, y_min=-1.12, y_max=1.12)
        Returns (r,g,b) with ints 0-255.
        """
        # Unpack with defaults
        if len(inp) < 4:
            raise ValueError("Input must contain at least (px,py,width,height)")
        px, py, width, height = inp[0], inp[1], inp[2], inp[3]
        max_iteration = inp[4] if len(inp) > 4 else 1000
        x_min = inp[5] if len(inp) > 5 else -2.0
        x_max = inp[6] if len(inp) > 6 else 0.47
        y_min = inp[7] if len(inp) > 7 else -1.12
        y_max = inp[8] if len(inp) > 8 else 1.12

        # Scale pixel to complex plane
        x0 = x_min + (px / (width - 1)) * (x_max - x_min)
        y0 = y_min + (py / (height - 1)) * (y_max - y_min)

        x = 0.0
        y = 0.0
        iteration = 0
        while (x * x + y * y <= 4.0) and (iteration < max_iteration):
            xtemp = x * x - y * y + x0
            y = 2 * x * y + y0
            x = xtemp
            iteration += 1

        # Map iteration to color using a simple palette: smooth gradient
        if iteration >= max_iteration:
            # inside the set -> black
            return (0, 0, 0)
        # normalize
        t = iteration / max_iteration
        # simple coloring: HSV-like gradient mapped to RGB
        r = int(9 * (1 - t) * t * t * t * 255)
        g = int(15 * (1 - t) * (1 - t) * t * t * 255)
        b = int(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255)
        # clamp
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        return (r, g, b)

    def get_num_outputs(self) -> int:
        # RGB output
        return 3

    def get_tailored_suite(self):
        # Return a small set of representative pixels (center, corners, edges)
        candidates = []
        width = 400
        height = 400
        points = [
            (0, 0),
            (width - 1, 0),
            (0, height - 1),
            (width - 1, height - 1),
            (width // 2, height // 2),
            (width // 4, height // 4),
            (3 * width // 4, 3 * height // 4),
        ]
        for (px, py) in points:
            candidates.append((px, py, width, height))
        # return as features, labels computed lazily by caller
        X, y = self.program_input_to_pandas(candidates)
        return X, y
