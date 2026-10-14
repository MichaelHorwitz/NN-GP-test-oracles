# GOAL: All the metrics at once
# RUN 10 independent tests on the whole pipeline and gather stats for all

from collections import Counter
import logging
import random

import numpy as np

from generation_methods.RandomSampler import RandomSampler
from generation_methods.LinearGeneration import LinearGeneration
from gp.GP import GeneticProgram
from gp.Node import (
    AddNode,
    LastOutput,
    Node,
    OperandNode,
    RandomNode,
    SubtractionNode,
    MultiplicationNode,
    DivisionNode,
    GreaterThanNode,
    LessThanNode,
    NotEqualNode,
    IfNode,
)
from neural_net.NeuralNet import NeuralNetwork
from software_under_test import SoftwareUnderTest
from software_under_test.Collatz import Collatz
from software_under_test.CreditCheck import CreditCheck
from software_under_test.TriangleClassifier import TriangleClassifier
import os
import matplotlib.pyplot as plt
from scipy import stats
import math
import pandas as pd

# Across Main
LOG_LEVEL = logging.INFO
TEST_SUITE_SIZE: int = 1000
NUM_INDEPENDANT_RUNS: int = 3

# GP
TERMINALS = [lambda: OperandNode(random.randint(-100, 100)), lambda: RandomNode()] # lambda: RandomNode(),
PRIMITIVES = [
    AddNode,
    SubtractionNode,
    MultiplicationNode,
    DivisionNode,
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
MAX_DEPTH = 5

# Neural Net
NUM_NEURONS=32
TEST_SIZE=0.15
ACTIVATION="relu"
EPOCHS=20
BATCH_SIZE=64
OPTIMIZER="adam"
VERBOSE=0
DROPOUT=None

def _average_histories(hist_list: list[dict]) -> dict:
    # hist_list: list of history dicts (or None). Return averaged history dict where each metric is averaged
    valid_hists = [h for h in hist_list if h]
    if not valid_hists:
        return {}
    # collect all metric keys (exclude val_ prefixed keys for separate handling)
    metric_keys = set()
    for h in valid_hists:
        for k in h.keys():
            if not k.startswith("val_"):
                metric_keys.add(k)

    averaged: dict = {}
    for metric in metric_keys:
        # gather series for metric across histories that contain it
        series_list = [np.array(h[metric]) for h in valid_hists if metric in h]
        if not series_list:
            continue
        # use minimum length to be safe
        min_len = min(s.shape[0] for s in series_list)
        trimmed = np.array([s[:min_len] for s in series_list], dtype=float)
        averaged[metric] = list(np.mean(trimmed, axis=0))
        # handle corresponding val_metric if present in at least one history
        val_metric = f"val_{metric}"
        val_series = [np.array(h[val_metric]) for h in valid_hists if val_metric in h]
        if val_series:
            min_len_val = min(s.shape[0] for s in val_series)
            trimmed_val = np.array([s[:min_len_val] for s in val_series], dtype=float)
            averaged[val_metric] = list(np.mean(trimmed_val, axis=0))
    return averaged
def _save_sut_histogram(
    branch_counters_by_generator: dict[str, Counter], sut_name: str
):
    all_branches = sorted(
        list(
            set(
                branch
                for counter in branch_counters_by_generator.values()
                for branch in counter.keys()
            )
        )
    )
    if not all_branches:
        print(
            f"No branches recorded for any generator on {sut_name}, skipping histogram."
        )
        return

    generator_names = list(branch_counters_by_generator.keys())

    x = np.arange(len(all_branches))
    width = 0.8 / len(generator_names)

    fig, ax = plt.subplots(figsize=(15, 7))

    for i, gen_name in enumerate(generator_names):
        offset = width * i - (width * (len(generator_names) - 1) / 2)
        counts = [
            branch_counters_by_generator[gen_name].get(branch, 0)
            for branch in all_branches
        ]
        ax.bar(x + offset, counts, width, label=gen_name)

    ax.set_ylabel("Frequency (hits across all runs and test cases)")
    ax.set_title(f"{sut_name} - Branch Frequencies by Generator")
    ax.set_xticks(x)
    ax.set_xticklabels(all_branches, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    fname = os.path.join(out_dir, f"hist_{sut_name}.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved histogram: {fname}")
def _save_nn_histogram(history, gen_name, sut_name):
    h = history
    metric_keys = [k for k in h.keys() if not k.startswith("val_")]
    for metric in metric_keys:
        plt.figure(figsize=(10, 5))
        ylabel = "Loss"
        if "accuracy" in metric:
            ylabel = "Accuracy"
        elif metric in ("mse", "mean_squared_error"):
            ylabel = "MSE"

        epochs = range(1, len(h[metric]) + 1)
        plt.plot(epochs, h[metric], label=f"{gen_name} {metric}")
        val_metric = f"val_{metric}"
        if val_metric in h:
            plt.plot(
                epochs,
                h[val_metric],
                linestyle="--",
                label=f"{gen_name} {val_metric}",
            )

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(
            f"{sut_name} - {gen_name} - {ylabel} per epoch (averaged over {NUM_INDEPENDANT_RUNS} runs)"
        )
        plt.legend()
        plt.tight_layout()
        metric_fname = os.path.join(
            out_dir, f"{sut_name}_{gen_name}_averaged_{metric}.png"
        )
        plt.savefig(metric_fname)
        plt.close()
        print(f"Saved averaged {ylabel} comparison for {gen_name}: {metric_fname}")
expected_branches_map = {
    "Collatz": {"RED", "ORANGE", "YELLOW", "GREEN", "BLUE", "PINK", "PURPLE"},
    "CreditCheck": {
        "red",
        "blue",
        "yellow",
        "green",
        "purple",
        "orange",
        "pink",
        "cyan",
        "brown",
        "gray",
        "lime",
        "navy",
        "maroon",
        "olive",
        "teal",
        "silver",
        "gold",
        "indigo",
        "violet",
        "coral",
        "khaki",
        "black",
        "white",
    },
    "TriangleClassifier": {"RED", "BLUE", "PINK", "GREEN", "ORANGE"},
}

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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

    prim_list: list[type[Node]] = PRIMITIVES
    term_list = TERMINALS

    out_dir = os.path.join(os.getcwd(), "charts")
    os.makedirs(out_dir, exist_ok=True)
    results_dir = os.path.join(os.getcwd(), "csv_results")
    os.makedirs(results_dir, exist_ok=True)
    results_csv = os.path.join(results_dir, "metrics.csv")
    # If file does not exist, write header
    if not os.path.exists(results_csv):
        with open(results_csv, "w", encoding="utf-8") as f:
            f.write(
                "sut,generator,mean_coverage,std_coverage,avg_loss,avg_val_loss,avg_mse,avg_val_mse,avg_accuracy,avg_val_accuracy\n"
            )

    generator_names = ["RandomSampler", "LinearGeneration", "GeneticProgram"]

    for sut_cls in [Collatz, CreditCheck, TriangleClassifier]:
    # for sut_cls in [TriangleClassifier]:
        cov_vals = {name: [] for name in generator_names}
        branch_counters = {name: Counter() for name in generator_names}
        means = {name: 0 for name in generator_names}
        std_devs = {name: 0 for name in generator_names}
        sut_name = sut_cls.__name__
        # Collect histories across independent runs for averaging
        histories_by_generator: dict[str, list] = {name: [] for name in generator_names}
        tailored = sut_cls().get_tailored_suite()
        for run_idx in range(NUM_INDEPENDANT_RUNS):
            sut: SoftwareUnderTest.SoftwareUnderTest = sut_cls()

            rng = RandomSampler(sut, test_suite_size=TEST_SUITE_SIZE)
            linear = LinearGeneration(sut, test_suite_size=TEST_SUITE_SIZE)
            gp = GeneticProgram(
                primative_list=prim_list,
                terminal_list=term_list, # type: ignore
                software_under_test=sut,
                pop_size=POP_SIZE,
                test_suite_size=TEST_SUITE_SIZE,
                cx_rate=CX_RATE,
                generations=NUM_GENERATIONS,
                max_depth=MAX_DEPTH,
                mut_rate=MUT_RATE,
                tournament_select_k=TOURNAMENT_SELECT_K,
            )
            generators = [rng, linear, gp]
            # generators = [gp]
            nns = []

            run_branch_sets: dict[str, set] = {name: set() for name in generator_names}

            for generator in generators:
                generator_name = generator.__class__.__name__
                generator.generate_test_suite()
                gen_df = pd.DataFrame(generator.test_suite)
                gen_df.to_csv(f"{sut_name}-{generator_name}-suite.csv")
                cov = sut.find_coverage(generator.test_suite)
                cov_vals[generator_name].append(cov)
                for test in generator.test_suite:
                    for br in sut.exec_sut_coverage(test):
                        branch_counters[generator_name][br] += 1
                        run_branch_sets[generator_name].add(br)
                nn = NeuralNetwork(
                    program_inputs=generator.test_suite,
                    software_under_test=sut,
                    num_neurons=NUM_NEURONS,
                    test_size=TEST_SIZE,
                    activation=ACTIVATION,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    optimizer=OPTIMIZER,
                    verbose=VERBOSE,
                    dropout=DROPOUT,
                    # learning_rate=0.01,
                )
                nn.genenrate_training_data()
                if (
                    sut_cls is TriangleClassifier
                    or sut_cls is Collatz
                    or sut_cls is CreditCheck
                ):
                    try:
                        # tailored = sut.get_tailored_suite()
                        # tailored may be either (list_of_inputs, list_of_outputs) or (pd.DataFrame, pd.DataFrame)
                        if isinstance(tailored, tuple) and len(tailored) == 2:
                            val_inputs, val_outputs = tailored
                            # If inputs are a list of domain objects (e.g., Applicant), convert to pandas via the SUT helper
                            if isinstance(val_inputs, list):
                                val_X, val_Y = sut.program_input_to_pandas(val_inputs)  # type: ignore
                            else:
                                # Assume the tailored suite already returned (features, labels)
                                val_X, val_Y = val_inputs, val_outputs
                        else:
                            # Unexpected format, fallback to default training behavior
                            raise ValueError("Unexpected tailored suite format")
                        _ = nn.train(validation_data=(val_X, val_Y))
                    except Exception:
                        # Fallback to default training behavior
                        _ = nn.train()
                else:
                    _ = nn.train()
                histories_by_generator[generator_name].append(
                    nn.last_history if hasattr(nn, "last_history") else None
                )
                nns.append(nn)

            for i, generator in enumerate(generators):
                other_suites = [
                    suite
                    for g in generators
                    if g != generator
                    for suite in g.test_suite
                ]
                # Tech debt if statement hell yeah baby
                if (
                    sut_cls is TriangleClassifier
                    or sut_cls is Collatz
                    or sut_cls is CreditCheck
                ):
                    try:
                        # tailored = sut.get_tailored_suite()
                        if isinstance(tailored, tuple) and len(tailored) == 2:
                            val_inputs, val_outputs = tailored
                            if isinstance(val_inputs, list):
                                val_X, val_Y = sut.program_input_to_pandas(val_inputs)  # type: ignore
                            else:
                                val_X, val_Y = val_inputs, val_outputs
                        else:
                            # Unexpected format, fallback to using other_suites
                            val_X, val_Y = sut.program_input_to_pandas(other_suites)  # type: ignore
                    except Exception:
                        val_X, val_Y = sut.program_input_to_pandas(other_suites)  # type: ignore
                else:
                    val_X, val_Y = sut.program_input_to_pandas(other_suites)  # type: ignore
                print(generator.__class__.__name__)
                nns[i].evaluate_model(val_X, val_Y)

            # After all generators have run for this independent run, print which branches were missed per generator
            print(f"Independent run {run_idx + 1} for SUT: {sut_name}")
            expected = expected_branches_map.get(sut_name)
            if expected is None:
                print(
                    f"  WARNING: No expected branch set found for {sut_name} in expected_branches_map"
                )
            for gen_name in generator_names:
                observed = run_branch_sets.get(gen_name, set())
                if expected is not None:
                    missed = expected - observed
                    extra = observed - expected
                    print(f"  Generator: {gen_name}")
                    print(
                        f"    Observed branches ({len(observed)}): {sorted(list(observed))}"
                    )
                    print(
                        f"    Missed expected branches ({len(missed)}): {sorted(list(missed))}"
                    )
                    if extra:
                        print(
                            f"    EXTRA branches observed not in expected map ({len(extra)}): {sorted(list(extra))}"
                        )
                else:
                    print(
                        f"  Generator: {gen_name} observed branches: {sorted(list(observed))}"
                    )

        averaged_histories = {
            name: _average_histories(histories_by_generator[name])
            for name in generator_names
        }

        # Save separate averaged plots per generator and per metric
        for gen_name in generator_names:
            h = averaged_histories.get(gen_name, {})
            if not h:
                continue
            hist_df: pd.DataFrame = pd.DataFrame(h)
            hist_df.to_csv(f"{results_dir}/{sut_name}-{gen_name}.csv")
            _save_nn_histogram(h, gen_name, sut_name)

        _save_sut_histogram(branch_counters, sut_name)

        # Overall validation: ensure expected_branches_map contains all observed branches
        overall_observed: set = set()
        for gen_counter in branch_counters.values():
            overall_observed.update(gen_counter.keys())

        expected = expected_branches_map.get(sut_name)
        if expected is None:
            print(f"WARNING: No expected branch set defined for SUT '{sut_name}'")
        else:
            missing_in_expected = overall_observed - expected
            missing_expected = expected - overall_observed
            if missing_in_expected:
                print(
                    f"WARNING: The expected_branches_map for '{sut_name}' is missing these observed branches: {sorted(list(missing_in_expected))}"
                )
            else:
                print(
                    f"All observed branches for '{sut_name}' are present in expected_branches_map."
                )
            if missing_expected:
                print(
                    f"NOTE: The expected_branches_map for '{sut_name}' contains branches not observed in any run: {sorted(list(missing_expected))}"
                )

        for name in generator_names:
            means[name] = np.mean(cov_vals[name])  # type: ignore
            std_devs[name] = np.std(cov_vals[name])  # type: ignore
            print(f"SUT: {sut_name}, Generator: {name}")
            print(f"  Mean Coverage: {means[name]}")
            print(f"  Std Dev Coverage: {std_devs[name]}")
        
        t_statistic, p_value_two_tailed = stats.ttest_ind(
            cov_vals['GeneticProgram'], 
            cov_vals['RandomSampler'], 
            equal_var=False,         
            alternative='greater'    
        )
        print("Comparing GP vs. Random Sampler:")
        print(f"  T-statistic: {t_statistic:.4f}")
        print(f"  P-value: {p_value_two_tailed:.4f}")
        alpha = 0.05
        if p_value_two_tailed < alpha: # type: ignore
            print(f"  Result: The p-value is less than {alpha}, so we reject the null hypothesis. (GP is better)")
        
        def _final_metric_avg(history: dict, metric_name: str):
            if not history or metric_name not in history:
                return float("nan")
            series = history[metric_name]
            try:
                if isinstance(series, (list, tuple, np.ndarray)) and len(series) > 0:
                    return float(series[-1])
            except Exception:
                pass
            return float("nan")

        with open(results_csv, "a", encoding="utf-8") as f:
            for gen_name in generator_names:
                h = averaged_histories.get(gen_name, {})
                avg_loss = _final_metric_avg(h, "loss")
                avg_val_loss = _final_metric_avg(h, "val_loss")
                avg_mse = _final_metric_avg(h, "mse")
                if math.isnan(avg_mse):
                    avg_mse = _final_metric_avg(h, "mean_squared_error")
                avg_val_mse = _final_metric_avg(h, "val_mse")
                if math.isnan(avg_val_mse):
                    avg_val_mse = _final_metric_avg(h, "val_mean_squared_error")
                avg_acc = _final_metric_avg(h, "accuracy")
                avg_val_acc = _final_metric_avg(h, "val_accuracy")

                # coverage mean/std
                cov_mean = means.get(gen_name, float("nan"))
                cov_std = std_devs.get(gen_name, float("nan"))

                # write CSV row
                f.write(
                    f"{sut_name},{gen_name},{cov_mean},{cov_std},{avg_loss},{avg_val_loss},{avg_mse},{avg_val_mse},{avg_acc},{avg_val_acc}\n"
                )
