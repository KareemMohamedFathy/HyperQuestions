"""
hyper_glad_algorithm.py

Hyper-GLAD, written in the same structure as your HyperMajorityVote.
Only the round-1 aggregation changes: instead of taking the most frequent
answer-tuple per hyper question (majority vote), GLAD estimates the winning
tuple. Data loading, hyper-question discovery, decoding, and the final
majority vote are kept the same as your code.

Depends only on glad.py (the multi-class, unsupervised GLAD). No hyper.py.

Returns (glad_baseline, final_hyper_glad), the GLAD analog of your
(majority_vote, final_hyper_mv), in the same dict format:
    { (dataset, question_number): (answer, frequency) }
so your plotting harness reads predictions[(dataset, q)][0] unchanged.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

from glad import glad


DATASETS = ["Chinese", "English", "IT", "Medicine", "Pokemon", "Science"]

GLAD_KW = dict(max_iter=40, m_steps=20, lr=0.02)

# Plain-GLAD baseline is independent of k and of the random hyper sampling,
# so compute it once per dataset.
_BASELINE = {}


def _encode(row, n_choices):
    """Tuple of answers -> single int code (base n_choices, low index first)."""
    code = 0
    radix = 1
    for a in row:
        code += int(a) * radix
        radix *= n_choices
    return code


def _decode(code, k, n_choices):
    """Int code -> tuple of k answers, matching the encode order."""
    out = []
    for _ in range(k):
        out.append(code % n_choices)
        code //= n_choices
    return tuple(out)


def _glad_baseline(data_set, df, n_choices):
    """Plain multi-class GLAD on the single questions (cached)."""
    if data_set not in _BASELINE:
        answers = df.values.astype(int)          # (questions, workers)
        pred, *_ = glad(answers, n_choices=n_choices, seed=0, **GLAD_KW)
        _BASELINE[data_set] = pred
    return _BASELINE[data_set]


def HyperGLAD(k, m, seed=None):
    rng = random.Random(seed)

    glad_baseline = {}
    questions_hyper_glad = {}     # (data_set, question_number, answer) -> votes

    for data_set in DATASETS:
        data_set_path = Path("datasets") / data_set / "answer.csv"
        df = pd.read_csv(data_set_path, header=None)
        questions_count = len(df)
        n_choices = int(df.values.max()) + 1     # choices per question (0..K-1)

        # ---- GLAD baseline on single questions (analog of normal MV) ----
        base_pred = _glad_baseline(data_set, df, n_choices)
        for q in range(questions_count):
            glad_baseline[(data_set, q)] = (int(base_pred[q]), 1)

        # ---- discover hyper questions (same shuffle + dedup as your code) ----
        seen_combinations = set()
        question_sets = []          # list of sorted k-tuples of question indices
        code_rows = []              # parallel list: per-worker int codes for each
        for _ in range(m):
            shuffled = list(range(questions_count))
            rng.shuffle(shuffled)
            for start in range(0, questions_count - k + 1, k):
                qset = tuple(sorted(shuffled[start:start + k]))
                if qset in seen_combinations:
                    continue
                seen_combinations.add(qset)

                # per-worker answers to this hyper question: (workers, k)
                worker_answers = df.iloc[list(qset)].T.values
                codes = np.array(
                    [_encode(worker_answers[w], n_choices)
                     for w in range(worker_answers.shape[0])],
                    dtype=np.int64,
                )
                question_sets.append(qset)
                code_rows.append(codes)

        # ===== round 1: GLAD instead of majority vote ====================
        # Build (n_hyper_questions, n_workers) code matrix and run GLAD over it.
        # K per hyper question = n_choices**k (product of constituent choices).
        code_matrix = np.vstack(code_rows)
        K_hyper = np.full(len(question_sets), n_choices ** k, dtype=np.int64)
        winners, *_ = glad(code_matrix, n_choices=K_hyper, seed=(seed or 0), **GLAD_KW)
        # =================================================================

        # ---- decode winners to per-question votes (same as your code) ----
        for qset, win_code in zip(question_sets, winners):
            answer_tuple = _decode(int(win_code), k, n_choices)
            for question_number, answer in zip(qset, answer_tuple):
                key = (data_set, question_number, int(answer))
                questions_hyper_glad[key] = questions_hyper_glad.get(key, 0) + 1

    # ---- final majority vote per question (identical to your code) ----
    final_hyper_glad = {}
    for (data_set, question, answer), frequency in questions_hyper_glad.items():
        key = (data_set, question)
        if key not in final_hyper_glad:
            final_hyper_glad[key] = (answer, frequency)
        else:
            _, current_freq = final_hyper_glad[key]
            if frequency > current_freq:
                final_hyper_glad[key] = (answer, frequency)

    return glad_baseline, final_hyper_glad


if __name__ == "__main__":
    HyperGLAD(4, 30)
