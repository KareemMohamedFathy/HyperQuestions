#!/usr/bin/env python3
"""
Annotator-bias scoring on the duck dataset.

Adapted from the optimism/pessimism idea in
Wich, Widmer, Hagerer & Groh (2021), "Investigating Annotator Bias in
Abusive Language Datasets" (RANLP 2021), turned into per-annotator SCORES
for a binary task.

Two approaches, exactly as specified:

  APPROACH 1 - prevalence deviation (UNSUPERVISED, no ground truth)
    For each annotator, how far their rate of "Yes" (label=1) sits from
    the crowd's average Yes-rate. Positive => says "Yes" more than the pack.

  APPROACH 2 - confusion vs a PSEUDO ground truth (Majority Vote / GLAD / Hyper-Questions)
    Estimate a pseudo ground truth, then for every annotator use only the
    two off-diagonal cells of a 2x2 confusion (bias) matrix:
        * pseudo-GT = "Yes"  but annotator said "No"  -> OPTIMISTIC  (o): misses positives / lenient
        * pseudo-GT = "No"   but annotator said "Yes" -> PESSIMISTIC (p): over-flags / strict
    Both are RATES, so they are comparable across annotators who labeled
    different numbers of items. We then report each annotator's deviation
    from the average o and p.

Data: reads labels.yaml (worker votes) and gt.yaml (ground truth) from a
local 'dataset/' folder. No download.

Label convention:  True  -> 1  ("Yes", the positive class)
                    False -> 0  ("No")
"""

import os
import sys

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    sys.exit("Need PyYAML:  pip install pyyaml")

EPS = 1e-9


# --------------------------------------------------------------------------- #
# Data loading  (local files only)
# --------------------------------------------------------------------------- #
def load_duck(folder="datasets"):
    """Return (long_df[worker,image,label], gt_series[image->0/1]).

    Reads labels.yaml + gt.yaml from a local folder (default: ./dataset).
    Searches: the given folder, ./datasets, then <script_dir>/dataset.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [folder, "dataset", os.path.join(here, "datasets"), here]
    src = next((c for c in candidates if c
                and os.path.exists(os.path.join(c, "labels.yaml"))
                and os.path.exists(os.path.join(c, "gt.yaml"))), None)
    if src is None:
        sys.exit("Could not find labels.yaml + gt.yaml. "
                 "Put both in a 'datasets/' folder next to this script.")

    print(f"[data] loading duck dataset from: {src}")
    with open(os.path.join(src, "labels.yaml")) as fh:
        labels = yaml.safe_load(fh)     # {worker: {image: bool}}
    with open(os.path.join(src, "gt.yaml")) as fh:
        gt = yaml.safe_load(fh)         # {image: bool}

    rows = []
    for worker, ann in labels.items():
        for image, lab in ann.items():
            rows.append((int(worker), int(image), int(bool(lab))))
    df = pd.DataFrame(rows, columns=["worker", "image", "label"])
    gt_s = pd.Series({int(k): int(bool(v)) for k, v in gt.items()}, name="gt")
    return df, gt_s


def to_matrix(df):
    """Annotator x item matrix of labels; NaN where an annotator didn't label."""
    return df.pivot(index="worker", columns="image", values="label")


# --------------------------------------------------------------------------- #
# Pseudo-ground-truth  (pluggable: any image->{0,1} mapping works)
# --------------------------------------------------------------------------- #
def majority_vote(mat):
    """Simple majority vote per item (ties -> 1). Returns image -> 0/1."""
    yes = mat.sum(axis=0)              # count of 1s per image
    n = mat.notna().sum(axis=0)        # votes per image
    return ((yes / n) >= 0.5).astype(int)

# NOTE: to use your own GLAD / Hyper-Questions instead, just build any
# Series/dict of  image -> {0,1}  and pass it to approach2_confusion(df, that).


# --------------------------------------------------------------------------- #
# APPROACH 1 - prevalence deviation (unsupervised)
# --------------------------------------------------------------------------- #
def approach1_prevalence(df):
    g = df.groupby("worker")["label"]
    yes_rate = g.mean()                       # fraction of "Yes" per annotator
    n = g.size()
    crowd_mean = yes_rate.mean()              # average Yes-rate across annotators
    dev = yes_rate - crowd_mean               # signed deviation
    score = dev / (yes_rate.std() + EPS)      # standardized score (z across annotators)

    out = pd.DataFrame({
        "n_labels": n,
        "yes_rate": yes_rate.round(3),
        "dev_from_mean": dev.round(3),
        "score_z": score.round(3),
    })
    out.attrs["crowd_mean_yes"] = crowd_mean
    return out.sort_values("score_z")


# --------------------------------------------------------------------------- #
# APPROACH 2 - two-cell confusion vs pseudo-GT  (optimism / pessimism)
# --------------------------------------------------------------------------- #
def approach2_confusion(df, pseudo_gt, x=3.0):
    """pseudo_gt: Series or dict image->{0,1}. Returns per-annotator o, p, groups.

    x: threshold for the group label. 'optimistic' if o > x*p,
       'pessimistic' if p > x*o, else 'medium'. This is the SAME x used by
       the later subgroup-filtering step, so it is parameterized here.
    """
    gt = pd.Series(pseudo_gt)
    d = df.copy()
    d["gt"] = d["image"].map(gt)
    d = d.dropna(subset=["gt"])
    d["gt"] = d["gt"].astype(int)

    recs = []
    for w, grp in d.groupby("worker"):
        pos = grp[grp["gt"] == 1]             # items whose (pseudo) truth is Yes
        neg = grp[grp["gt"] == 0]             # items whose (pseudo) truth is No
        # OPTIMISTIC o = P(annotator says No | truth Yes)  -> misses positives
        o = (pos["label"] == 0).mean() if len(pos) else np.nan
        # PESSIMISTIC p = P(annotator says Yes | truth No) -> over-flags
        p = (neg["label"] == 1).mean() if len(neg) else np.nan
        recs.append((w, len(pos), len(neg), o, p))

    res = pd.DataFrame(recs, columns=["worker", "n_pos", "n_neg", "o", "p"]).set_index("worker")
    mean_o, mean_p = res.o.mean(), res.p.mean()
    res["o_dev"] = res.o - mean_o             # deviation from average optimism
    res["p_dev"] = res.p - mean_p             # deviation from average pessimism

    # grouping: dominating direction by factor x, else "medium"
    def grp_label(r):
        if r.p > x * r.o:
            return "pessimistic"
        if r.o > x * r.p:
            return "optimistic"
        return "medium"
    res["group"] = res.apply(grp_label, axis=1)

    res.attrs["mean_o"] = mean_o
    res.attrs["mean_p"] = mean_p
    return res.round(3)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # optional folder arg; defaults to ./dataset
    folder = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    df, gt = load_duck(folder)

    n_workers, n_items = df.worker.nunique(), df.image.nunique()
    print(f"duck dataset: {n_workers} annotators, {n_items} items, {len(df)} labels")
    print(f"True prevalence of 'Yes' (real GT): {gt.mean():.3f}\n")

    # ---------- APPROACH 1 ----------
    a1 = approach1_prevalence(df)
    print("=" * 70)
    print("APPROACH 1 - prevalence deviation (unsupervised)")
    print(f"crowd mean Yes-rate = {a1.attrs['crowd_mean_yes']:.3f}")
    print("(score_z < 0 => says 'No' more than average; > 0 => says 'Yes' more)")
    print("-" * 70)
    print(a1.head(6).to_string())
    print(" ...")
    print(a1.tail(6).to_string())

    # ---------- APPROACH 2 ----------
    pseudo_gt = majority_vote(to_matrix(df))     # swap in GLAD / Hyper-Questions here
    a2 = approach2_confusion(df, pseudo_gt, x=3)
    print("\n" + "=" * 70)
    print("APPROACH 2 - optimism/pessimism vs MAJORITY-VOTE pseudo-GT")
    print(f"mean optimistic o = {a2.attrs['mean_o']:.3f}   "
          f"mean pessimistic p = {a2.attrs['mean_p']:.3f}")
    print("  o = P(says No | truth Yes)  |  p = P(says Yes | truth No)")
    print("-" * 70)
    print(a2.sort_values("p", ascending=False)
            [["n_pos", "n_neg", "o", "p", "o_dev", "p_dev", "group"]]
            .head(10).to_string())
    print("\ngroup counts:", a2.group.value_counts().to_dict())

    # ---------- sanity: does the pseudo-GT scoring track the real-GT scoring? ----------
    a2_true = approach2_confusion(df, gt, x=3)
    comp = pd.DataFrame({"o_mv": a2.o, "o_true": a2_true.o,
                         "p_mv": a2.p, "p_true": a2_true.p}).dropna()
    print("\n" + "=" * 70)
    print("SANITY CHECK - pseudo-GT (MV) scores vs real-GT scores")
    print(f"  corr(optimistic o):  {comp.o_mv.corr(comp.o_true):.3f}")
    print(f"  corr(pessimistic p): {comp.p_mv.corr(comp.p_true):.3f}")
    agree = (pseudo_gt.reindex(gt.index) == gt).mean()
    print(f"\nMV pseudo-GT vs real GT agreement: {agree*100:.1f}%")