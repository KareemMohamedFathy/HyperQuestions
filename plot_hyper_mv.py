"""
plot_hyper_mv_v2.py

Plots for your existing HyperMajorityVote(k, m) function.

This version creates:

GROUP 1:
    One plot per dataset/category comparing:
        Normal Majority Voting vs Hyper-MV for different k values.

GROUP 2:
    One combined plot showing:
        Hyper-MV accuracy across k for every dataset/category.

Truth format:
    datasets/Chinese/truth.csv
    datasets/English/truth.csv
    ...
Each truth.csv must be one column, one correct answer per question.

Important:
    Your HyperMajorityVote(k, m) function must return:
        return majority_vote, final_hyper_mv

This script can load your algorithm file even if it is named hyper-questions.py.
Human filename choices, what a rich source of suffering.
"""

import importlib.util
import os
import statistics
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------
# Settings
# -----------------------------

DATASETS = ["Chinese", "English", "IT", "Medicine", "Pokemon", "Science"]

DATASETS_DIR = Path("datasets")

# It will try these in order.
# Keep your code in one of these filenames.
ALGORITHM_FILE_CANDIDATES = [
    Path("hyper_questions.py"),
    Path("hyper-questions.py"),
    Path("hyper_mv.py"),
]

K_VALUES = [2, 3, 4, 5, 6, 7,8,9]

# This is your m parameter.
M = 30

# Set to 1 if you want one run only.
# Keep 10 if you want mean/std over random sampled hyper-questions.
REPEATS = 10

# Your HyperMajorityVote prints giant dictionaries.
# Keep this False unless you enjoy console garbage fires.
SHOW_ALGORITHM_OUTPUT = False

OUTPUT_DIR = Path("plots")


# -----------------------------
# Load your existing function
# -----------------------------

def load_hyper_majority_vote_function():
    algorithm_file = None

    for candidate in ALGORITHM_FILE_CANDIDATES:
        if candidate.exists():
            algorithm_file = candidate
            break

    if algorithm_file is None:
        searched = ", ".join(str(path) for path in ALGORITHM_FILE_CANDIDATES)
        raise FileNotFoundError(
            f"Could not find your algorithm file. Looked for: {searched}"
        )

    spec = importlib.util.spec_from_file_location("hyper_algorithm_module", algorithm_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "HyperMajorityVote"):
        raise AttributeError(
            f"{algorithm_file} exists, but it does not contain a function named "
            f"HyperMajorityVote."
        )

    print(f"Loaded HyperMajorityVote from {algorithm_file}")
    return module.HyperMajorityVote


HyperMajorityVote = load_hyper_majority_vote_function()


# -----------------------------
# Truth and accuracy
# -----------------------------

def load_truth(dataset):
    truth_path = DATASETS_DIR / dataset / "truth.csv"

    if not truth_path.exists():
        raise FileNotFoundError(
            f"Missing truth file for {dataset}: {truth_path}\n"
            f"Expected format: one column, one correct answer per question."
        )

    truth = pd.read_csv(truth_path, header=None).iloc[:, 0].tolist()
    return truth


def calculate_accuracy(predictions, truth, dataset):
    correct = 0
    total = len(truth)

    missing_predictions = []

    for question_number, true_answer in enumerate(truth):
        key = (dataset, question_number)

        if key not in predictions:
            missing_predictions.append(question_number)
            continue

        predicted_answer = predictions[key][0]

        # Compare as strings so int 3 and "3" do not start a tiny civil war.
        if str(predicted_answer) == str(true_answer):
            correct += 1

    if missing_predictions:
        print(
            f"Warning: {dataset} has missing predictions for questions: "
            f"{missing_predictions}"
        )

    return correct / total


def call_algorithm(k, m):
    if SHOW_ALGORITHM_OUTPUT:
        result = HyperMajorityVote(k, m)
    else:
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull):
                result = HyperMajorityVote(k, m)

    if result is None:
        raise ValueError(
            "HyperMajorityVote returned None. Add this at the end of your function:\n"
            "    return majority_vote, final_hyper_mv"
        )

    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(
            "HyperMajorityVote must return exactly two things:\n"
            "    return majority_vote, final_hyper_mv"
        )

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

            majority_vote, final_hyper_mv = call_algorithm(k, M)

            for dataset in DATASETS:
                mv_accuracy = calculate_accuracy(
                    predictions=majority_vote,
                    truth=truths[dataset],
                    dataset=dataset,
                )

                hyper_accuracy = calculate_accuracy(
                    predictions=final_hyper_mv,
                    truth=truths[dataset],
                    dataset=dataset,
                )

                rows.append({
                    "dataset": dataset,
                    "k": k,
                    "repeat": repeat,
                    "majority_vote_accuracy": mv_accuracy,
                    "hyper_mv_accuracy": hyper_accuracy,
                })

    results = pd.DataFrame(rows)
    results.to_csv("hyper_mv_raw_results.csv", index=False)

    summary = (
        results
        .groupby(["dataset", "k"], as_index=False)
        .agg(
            majority_vote_accuracy=("majority_vote_accuracy", "mean"),
            hyper_mv_mean_accuracy=("hyper_mv_accuracy", "mean"),
            hyper_mv_std_accuracy=("hyper_mv_accuracy", "std"),
        )
    )

    summary["hyper_mv_std_accuracy"] = summary["hyper_mv_std_accuracy"].fillna(0)
    summary.to_csv("hyper_mv_summary.csv", index=False)

    print("Saved hyper_mv_raw_results.csv")
    print("Saved hyper_mv_summary.csv")

    return summary


# -----------------------------
# GROUP 1 plots:
# One plot per dataset comparing MV vs Hyper-MV over k
# -----------------------------

def plot_group_1_per_dataset_comparison(summary):
    group_dir = OUTPUT_DIR / "group_1_per_dataset_comparison"
    group_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        subset = summary[summary["dataset"] == dataset].sort_values("k")

        ks = subset["k"].tolist()
        hyper_means = subset["hyper_mv_mean_accuracy"].tolist()
        hyper_stds = subset["hyper_mv_std_accuracy"].tolist()
        mv_accuracy = subset["majority_vote_accuracy"].iloc[0]

        labels = ["MV"] + [f"k={k}" for k in ks]
        values = [mv_accuracy] + hyper_means
        errors = [0] + hyper_stds

        plt.figure(figsize=(9, 5))
        plt.bar(labels, values, yerr=errors, capsize=5)

        plt.title(f"{dataset}: Majority Voting vs Hyper-MV")
        plt.xlabel("Method")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        output_file = group_dir / f"{dataset}_mv_vs_hyper_mv.png"
        plt.savefig(output_file, dpi=300)
        plt.show()
        plt.close()

        print(f"Saved {output_file}")


# -----------------------------
# GROUP 2 plot:
# One combined graph showing every dataset's Hyper-MV accuracy over k
# -----------------------------

def plot_group_2_all_categories_by_k(summary):
    group_dir = OUTPUT_DIR / "group_2_all_categories_by_k"
    group_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for dataset in DATASETS:
        subset = summary[summary["dataset"] == dataset].sort_values("k")

        plt.errorbar(
            subset["k"],
            subset["hyper_mv_mean_accuracy"],
            yerr=subset["hyper_mv_std_accuracy"],
            marker="o",
            capsize=4,
            label=dataset,
        )

    plt.title("Hyper-MV Accuracy by k for Each Category")
    plt.xlabel("Hyper-question size k")
    plt.ylabel("Accuracy")
    plt.xticks(K_VALUES)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_file = group_dir / "all_categories_hyper_mv_by_k.png"
    plt.savefig(output_file, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved {output_file}")


# Optional extra:
# One combined graph showing MV baseline per dataset/category.
def plot_majority_vote_baseline_by_category(summary):
    group_dir = OUTPUT_DIR / "group_2_all_categories_by_k"
    group_dir.mkdir(parents=True, exist_ok=True)

    baseline = (
        summary
        .groupby("dataset", as_index=False)
        .agg(majority_vote_accuracy=("majority_vote_accuracy", "first"))
    )

    plt.figure(figsize=(9, 5))
    plt.bar(baseline["dataset"], baseline["majority_vote_accuracy"])

    plt.title("Normal Majority Voting Accuracy by Category")
    plt.xlabel("Category")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_file = group_dir / "majority_vote_accuracy_by_category.png"
    plt.savefig(output_file, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved {output_file}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    summary = run_experiments()

    plot_group_1_per_dataset_comparison(summary)
    plot_group_2_all_categories_by_k(summary)
    plot_majority_vote_baseline_by_category(summary)


if __name__ == "__main__":
    main()
