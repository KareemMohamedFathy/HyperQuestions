"""
plot_hyper_glad.py

Hyper-GLAD version of your plot_hyper_mv_v2.py. Same structure, same plots,
GLAD swapped in for Majority Voting.

GROUP 1: one plot per dataset comparing GLAD vs Hyper-GLAD across k.
GROUP 2: one combined plot of Hyper-GLAD accuracy across k for every dataset.

Needs, in the same folder:
    hyper_glad_algorithm.py   (defines HyperGLAD)
    glad.py, hyper.py
    datasets/<Dataset>/answer.csv  and  truth.csv

Note on cost: GLAD runs EM per hyper-question set, so this is much slower than
Hyper-MV. Keep K_VALUES small (the paper only used k=2..4 for Hyper-GLAD) and
REPEATS modest. k >= 6 on the 20-question datasets is slow and tends to get
worse, not better -- experts rarely match on huge hyper questions.
"""

import os
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hyper_glad_algorithm import HyperGLAD, DATASETS


# -----------------------------
# Settings
# -----------------------------
DATASETS_DIR = Path("datasets")

# Hyper-GLAD is well-behaved for small k; the paper used 2..4.
K_VALUES = [2, 3, 4, 5]

# Number of random permutations for hyper-question sampling (your m).
M = 30

# Mean/std over this many random hyper-question samples.
REPEATS = 5

SHOW_ALGORITHM_OUTPUT = False

OUTPUT_DIR = Path("plots_glad")


# -----------------------------
# Truth and accuracy (identical to your harness)
# -----------------------------
def load_truth(dataset):
    truth_path = DATASETS_DIR / dataset / "truth.csv"
    if not truth_path.exists():
        raise FileNotFoundError(
            f"Missing truth file for {dataset}: {truth_path}\n"
            f"Expected format: one column, one correct answer per question."
        )
    return pd.read_csv(truth_path, header=None).iloc[:, 0].tolist()


def calculate_accuracy(predictions, truth, dataset):
    correct = 0
    total = len(truth)
    missing = []
    for question_number, true_answer in enumerate(truth):
        key = (dataset, question_number)
        if key not in predictions:
            missing.append(question_number)
            continue
        if str(predictions[key][0]) == str(true_answer):
            correct += 1
    if missing:
        print(f"Warning: {dataset} has missing predictions for: {missing}")
    return correct / total


def call_algorithm(k, m, seed):
    if SHOW_ALGORITHM_OUTPUT:
        result = HyperGLAD(k, m, seed=seed)
    else:
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull):
                result = HyperGLAD(k, m, seed=seed)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("HyperGLAD must return (glad_baseline, final_hyper_glad)")
    return result


# -----------------------------
# Run experiments
# -----------------------------
def run_experiments():
    truths = {dataset: load_truth(dataset) for dataset in DATASETS}
    rows = []

    for k in K_VALUES:
        print(f"Running k={k}")
        for repeat in range(REPEATS):
            print(f"  repeat {repeat + 1}/{REPEATS}")
            glad_baseline, final_hyper_glad = call_algorithm(k, M, seed=repeat)

            for dataset in DATASETS:
                glad_acc = calculate_accuracy(glad_baseline, truths[dataset], dataset)
                hyper_acc = calculate_accuracy(final_hyper_glad, truths[dataset], dataset)
                rows.append({
                    "dataset": dataset,
                    "k": k,
                    "repeat": repeat,
                    "glad_accuracy": glad_acc,
                    "hyper_glad_accuracy": hyper_acc,
                })

    results = pd.DataFrame(rows)
    results.to_csv("hyper_glad_raw_results.csv", index=False)

    summary = (
        results
        .groupby(["dataset", "k"], as_index=False)
        .agg(
            glad_accuracy=("glad_accuracy", "mean"),
            hyper_glad_mean_accuracy=("hyper_glad_accuracy", "mean"),
            hyper_glad_std_accuracy=("hyper_glad_accuracy", "std"),
        )
    )
    summary["hyper_glad_std_accuracy"] = summary["hyper_glad_std_accuracy"].fillna(0)
    summary.to_csv("hyper_glad_summary.csv", index=False)

    print("Saved hyper_glad_raw_results.csv")
    print("Saved hyper_glad_summary.csv")
    return summary


# -----------------------------
# GROUP 1: per-dataset GLAD vs Hyper-GLAD over k
# -----------------------------
def plot_group_1_per_dataset_comparison(summary):
    group_dir = OUTPUT_DIR / "group_1_per_dataset_comparison"
    group_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        subset = summary[summary["dataset"] == dataset].sort_values("k")
        ks = subset["k"].tolist()
        hyper_means = subset["hyper_glad_mean_accuracy"].tolist()
        hyper_stds = subset["hyper_glad_std_accuracy"].tolist()
        glad_accuracy = subset["glad_accuracy"].iloc[0]

        labels = ["GLAD"] + [f"k={k}" for k in ks]
        values = [glad_accuracy] + hyper_means
        errors = [0] + hyper_stds

        plt.figure(figsize=(9, 5))
        plt.bar(labels, values, yerr=errors, capsize=5)
        plt.title(f"{dataset}: GLAD vs Hyper-GLAD")
        plt.xlabel("Method")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        output_file = group_dir / f"{dataset}_glad_vs_hyper_glad.png"
        plt.savefig(output_file, dpi=300)
        plt.show()
        plt.close()
        print(f"Saved {output_file}")


# -----------------------------
# GROUP 2: all datasets' Hyper-GLAD accuracy over k
# -----------------------------
def plot_group_2_all_categories_by_k(summary):
    group_dir = OUTPUT_DIR / "group_2_all_categories_by_k"
    group_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for dataset in DATASETS:
        subset = summary[summary["dataset"] == dataset].sort_values("k")
        plt.errorbar(
            subset["k"],
            subset["hyper_glad_mean_accuracy"],
            yerr=subset["hyper_glad_std_accuracy"],
            marker="o", capsize=4, label=dataset,
        )

    plt.title("Hyper-GLAD Accuracy by k for Each Category")
    plt.xlabel("Hyper-question size k")
    plt.ylabel("Accuracy")
    plt.xticks(K_VALUES)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_file = group_dir / "all_categories_hyper_glad_by_k.png"
    plt.savefig(output_file, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved {output_file}")


# Optional: GLAD baseline per dataset.
def plot_glad_baseline_by_category(summary):
    group_dir = OUTPUT_DIR / "group_2_all_categories_by_k"
    group_dir.mkdir(parents=True, exist_ok=True)

    baseline = (
        summary.groupby("dataset", as_index=False)
        .agg(glad_accuracy=("glad_accuracy", "first"))
    )

    plt.figure(figsize=(9, 5))
    plt.bar(baseline["dataset"], baseline["glad_accuracy"])
    plt.title("GLAD Accuracy by Category")
    plt.xlabel("Category")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_file = group_dir / "glad_accuracy_by_category.png"
    plt.savefig(output_file, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved {output_file}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = run_experiments()
    plot_group_1_per_dataset_comparison(summary)
    plot_group_2_all_categories_by_k(summary)
    plot_glad_baseline_by_category(summary)


if __name__ == "__main__":
    main()
