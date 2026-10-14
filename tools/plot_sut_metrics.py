"""Generate combined metric + validation metric charts for SUTs that have all three generators.

Reads csv_results/metrics.csv and produces PNG charts into charts/ and charts_stable/.

Dependencies: pandas, matplotlib

Usage: python tools/plot_sut_metrics.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "csv_results" / "metrics.csv"
OUT_DIRS = [ROOT / "charts", ROOT / "charts_stable"]

METRIC_PAIRS = [
    ("loss", "val_loss", "Loss"),
    ("mse", "val_mse", "MSE"),
    ("accuracy", "val_accuracy", "Accuracy"),
]
GENERATORS = ["RandomSampler", "LinearGeneration", "GeneticProgram"]


def ensure_dirs():
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def load_metrics():
    if not CSV.exists():
        raise FileNotFoundError(f"Metrics CSV not found at: {CSV}")
    df = pd.read_csv(CSV)
    return df


def suts_with_all_generators(df):
    groups = df.groupby("sut")["generator"].nunique()
    suts = groups[groups >= 3].index.tolist()
    return suts


def plot_for_sut(df, sut):
    """For a given SUT, read per-generator per-epoch CSV files and plot metrics over epochs.

    Solid lines = training metric, dashed lines = validation metric.
    One plot per metric (Loss, MSE, Accuracy). Lines are colored by generator.
    """
    for (metric, val_metric, label) in METRIC_PAIRS:
        plt.figure(figsize=(14.42, 4.67 ))
        plotted_any = False
        for gen in GENERATORS:
            csv_path = ROOT / "csv_results" / f"{sut}-{gen}.csv"
            if not csv_path.exists():
                # skip missing generator files
                continue
            gdf = pd.read_csv(csv_path, index_col=0)
            # epochs are the index; ensure numeric
            try:
                epochs = gdf.index.astype(int)
            except Exception:
                epochs = range(len(gdf))

            if metric in gdf.columns:
                plt.plot(epochs, gdf[metric].values, label=f"{gen}")
                plotted_any = True
            if val_metric in gdf.columns:
                plt.plot(epochs, gdf[val_metric].values, linestyle="--", label=f"{gen} val")
                plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.title(f"{sut} - {label} over epochs")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()

        for outdir in OUT_DIRS:
            outpath = outdir / f"{sut}-{label}-epochs.png"
            plt.savefig(outpath)
        plt.close()


def main():
    ensure_dirs()
    df = load_metrics()
    suts = suts_with_all_generators(df)
    if not suts:
        print("No SUTs found with all three generators. Nothing to plot.")
        return
    print(f"Found SUTs with all generators: {suts}")
    for sut in suts:
        print(f"Plotting metrics for: {sut}")
        plot_for_sut(df, sut)
    print("Charts generated in:")
    for d in OUT_DIRS:
        print(f" - {d}")


if __name__ == "__main__":
    main()
