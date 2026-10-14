"""
Test to determine the minimum test suite size needed to achieve 100% path coverage
for each generation method, with a cap at 10,000 test suites.
"""

import logging
import os
import random
from typing import Any, Dict

import pandas as pd

from generation_methods.RandomSampler import RandomSampler
from generation_methods.LinearGeneration import LinearGeneration
from gp.GP import GeneticProgram
from gp.Node import (
    AddNode,
    LastOutput,
    OperandNode,
    GreaterThanNode,
    LessThanNode,
    NotEqualNode,
    IfNode,
    SubtractionNode,
)
from software_under_test.Collatz import Collatz
from software_under_test.CreditCheck import CreditCheck
from software_under_test.TriangleClassifier import TriangleClassifier
from software_under_test.SoftwareUnderTest import SoftwareUnderTest


# Configuration
LOG_LEVEL = logging.INFO
MAX_SUITE_COUNT = 40  # Cap at 10,000 test suites
NUM_INDEPENDENT_RUNS = 10  # Number of independent runs for statistical reliability
SUITE_SIZE_INCREMENT = 10  # How much to increase suite size each iteration

# GP Configuration
TERMINALS = [lambda: OperandNode(random.randint(-1, 257)), lambda: OperandNode(128)]
PRIMITIVES = [
    AddNode,
    SubtractionNode,
    GreaterThanNode,
    LessThanNode,
    NotEqualNode,
    IfNode,
    LastOutput,
]
POP_SIZE = 100
NUM_GENERATIONS = 21
TOURNAMENT_SELECT_K = 3
MUT_RATE = 0.4
CX_RATE = 0.7
MAX_DEPTH = 3


def setup_logging():
    """Configure logging for the experiment"""
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.NOTSET)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def find_min_suite_size_for_full_coverage(
    generator_class,
    sut: SoftwareUnderTest,
    run_idx: int,
    generator_name: str,
    sut_name: str,
) -> Dict[str, Any]:
    """
    Find the minimum test suite size needed to achieve 100% coverage.
    
    Returns a dict with:
        - 'suite_size': the size that achieved 100% coverage (or MAX_SUITE_COUNT if not achieved)
        - 'coverage': the final coverage percentage
        - 'achieved_100': boolean indicating if 100% was achieved
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"Run {run_idx + 1}: Testing {generator_name} on {sut_name}"
    )
    
    current_size = SUITE_SIZE_INCREMENT
    coverage = 0.0  # Initialize coverage variable
    
    while current_size <= MAX_SUITE_COUNT:
        # Create generator with current test suite size
        if generator_class == RandomSampler:
            generator = RandomSampler(sut, test_suite_size=current_size)
        elif generator_class == LinearGeneration:
            generator = LinearGeneration(sut, test_suite_size=current_size)
        elif generator_class == GeneticProgram:
            generator = GeneticProgram(
                primative_list=PRIMITIVES,
                terminal_list=TERMINALS,  # type: ignore
                software_under_test=sut,
                pop_size=POP_SIZE,
                test_suite_size=current_size,
                cx_rate=CX_RATE,
                generations=NUM_GENERATIONS,
                max_depth=MAX_DEPTH,
                mut_rate=MUT_RATE,
                tournament_select_k=TOURNAMENT_SELECT_K,
            )
        else:
            raise ValueError(f"Unknown generator class: {generator_class}")
        
        # Generate test suite and measure coverage
        generator.generate_test_suite()
        coverage = sut.find_coverage(generator.test_suite)
        
        logger.debug(
            f"  Size {current_size}: Coverage = {coverage:.2%}"
        )
        
        # Check if we achieved 100% coverage
        if coverage >= 1.0:
            logger.info(
                f"  ✓ Achieved 100% coverage with suite size: {current_size}"
            )
            return {
                'suite_size': current_size,
                'coverage': coverage,
                'achieved_100': True
            }
        
        # Increment suite size
        current_size += SUITE_SIZE_INCREMENT
    
    # If we reached here, we didn't achieve 100% coverage
    logger.warning(
        f"  ✗ Failed to achieve 100% coverage. Max coverage at size {MAX_SUITE_COUNT}: {coverage:.2%}"
    )
    return {
        'suite_size': MAX_SUITE_COUNT,
        'coverage': coverage,
        'achieved_100': False
    }


def run_experiment():
    """Main experiment function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Prepare output directory
    out_dir = os.path.join(os.getcwd(), "coverage_results")
    os.makedirs(out_dir, exist_ok=True)
    
    # Define test configurations
    # sut_classes = [Collatz, CreditCheck, TriangleClassifier]
    sut_classes = [CreditCheck]
    generator_configs = [
        # ("RandomSampler", RandomSampler),
        # ("LinearGeneration", LinearGeneration),
        ("GeneticProgram", GeneticProgram),
    ]
    
    # Store all results
    all_results = []
    
    # Run experiments
    for sut_cls in sut_classes:
        sut_name = sut_cls.__name__
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing SUT: {sut_name}")
        logger.info(f"{'='*60}\n")
        
        for gen_name, gen_class in generator_configs:
            logger.info(f"\nGenerator: {gen_name}")
            logger.info(f"{'-'*40}")
            
            # Run multiple independent trials
            run_results = []
            for run_idx in range(NUM_INDEPENDENT_RUNS):
                # Create fresh SUT instance for each run
                sut = sut_cls()
                
                # Find minimum suite size for this run
                result = find_min_suite_size_for_full_coverage(
                    gen_class, sut, run_idx, gen_name, sut_name
                )
                
                run_results.append(result)
                
                # Store result with metadata
                all_results.append({
                    'sut': sut_name,
                    'generator': gen_name,
                    'run': run_idx + 1,
                    'suite_size': result['suite_size'],
                    'coverage': result['coverage'],
                    'achieved_100': result['achieved_100']
                })
            
            # Calculate statistics for this generator on this SUT
            sizes = [r['suite_size'] for r in run_results]
            coverages = [r['coverage'] for r in run_results]
            success_count = sum(1 for r in run_results if r['achieved_100'])
            
            logger.info(f"\n  Summary for {gen_name} on {sut_name}:")
            logger.info(f"    Runs achieving 100%: {success_count}/{NUM_INDEPENDENT_RUNS}")
            logger.info(f"    Average suite size: {sum(sizes)/len(sizes):.1f}")
            logger.info(f"    Min suite size: {min(sizes)}")
            logger.info(f"    Max suite size: {max(sizes)}")
            logger.info(f"    Average coverage: {sum(coverages)/len(coverages):.2%}")
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    output_file = os.path.join(out_dir, "coverage_suite_size_results.csv")
    results_df.to_csv(output_file, index=False)
    logger.info(f"\n{'='*60}")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"{'='*60}\n")
    
    # Create summary statistics
    summary_stats = []
    for sut_name in [cls.__name__ for cls in sut_classes]:
        for gen_name, _ in generator_configs:
            subset = results_df[
                (results_df['sut'] == sut_name) & 
                (results_df['generator'] == gen_name)
            ]
            
            summary_stats.append({
                'sut': sut_name,
                'generator': gen_name,
                'success_rate': subset['achieved_100'].mean(),
                'avg_suite_size': subset['suite_size'].mean(),
                'min_suite_size': subset['suite_size'].min(),
                'max_suite_size': subset['suite_size'].max(),
                'std_suite_size': subset['suite_size'].std(),
                'avg_coverage': subset['coverage'].mean(),
            })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_file = os.path.join(out_dir, "coverage_suite_size_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Summary statistics saved to: {summary_file}\n")
    
    # Print summary table
    logger.info("\nSummary Table:")
    logger.info(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_experiment()
