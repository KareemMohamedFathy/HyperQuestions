"""
Multi-class GLAD (Generative model of Labels, Abilities, and Difficulties).

"""

import numpy as np


def _sigmoid(x):
    # numerically stable
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def glad(answers, n_choices, max_iter=50, m_steps=25, lr=0.01,
         prior_alpha=(1.0, 1.0), prior_logbeta=(1.0, 1.0), tol=1e-5,
         seed=0):
    """
    Parameters
    ----------
    answers : 2D int array, shape (n_questions, n_workers)
        answers[j, i] = choice (0..n_choices[j]-1) worker i gave to question j,
        or -1 for "did not answer".
    n_choices : int or 1D int array of length n_questions
        number of candidate choices per question (K_j). For hyper questions
        this is the product of the constituent single-question choice counts.
    Returns
    -------
    pred : 1D int array, the argmax true label per question
    posterior : (n_questions, max_K) array of label posteriors (ragged, padded)
    alpha : worker abilities
    beta  : question 1/difficulty (= exp(b))
    """
    rng = np.random.default_rng(seed)
    answers = np.asarray(answers)
    n_q, n_w = answers.shape

    K = (np.full(n_q, int(n_choices)) if np.isscalar(n_choices)
         else np.asarray(n_choices, dtype=int))

    mu_a, sd_a = prior_alpha
    mu_b, sd_b = prior_logbeta

    # the distinct answers actually observed.
    candidates = [np.unique(answers[j]) for j in range(n_q)]

    # init
    alpha = np.full(n_w, 1.0) + 0.01 * rng.standard_normal(n_w)
    b = np.full(n_q, 1.0) + 0.01 * rng.standard_normal(n_q)   # log-beta

    pred = np.array([c[0] if len(c) else 0 for c in candidates])
    post_at_answer = np.zeros((n_q, n_w))  # p_ij used by the M-step
    prev_ll = -np.inf

    for _ in range(max_iter):
        beta = np.exp(b)

        # E-step 
        x = alpha[None, :] * beta[:, None]                 # (n_q, n_w) = a_i b_j
        s = _sigmoid(x)
        log_correct = np.log(np.clip(s, 1e-12, 1.0)) #probablity of correctness
        log_wrong = (np.log(np.clip(1.0 - s, 1e-12, 1.0))
                     - np.log(np.maximum(K[:, None] - 1, 1)))   # depends on K_j

        ll = 0.0
        for j in range(n_q): # loop on questions
            cand = candidates[j]
            ans_j = answers[j]
            logp = np.empty(len(cand))
            ans_j = answers[j]

            for ci, c in enumerate(cand): # loop on answers
                hit = ans_j == c
                miss = ~hit
                logp[ci] = log_correct[j, hit].sum() + log_wrong[j, miss].sum() 
            mmax = logp.max()
            w = np.exp(logp - mmax)
            Z = w.sum()
            postj = w / Z #posterior for each 
            ll += mmax + np.log(Z)
            pred[j] = cand[postj.argmax()]
            # map posterior back onto each answering worker's actual choice
            # p_ij = posterior mass on the label equal to what worker i said
            cand_to_post = {int(c): postj[ci] for ci, c in enumerate(cand)}
            row = post_at_answer[j]
            row[:] = 0.0
            post_at_answer[j] = np.array([ # worker ans is probably correct
            cand_to_post[int(ans_j[i])]
            for i in range(n_w)
            ])
            
        # M-step 
        for _ms in range(m_steps):
            beta = np.exp(b)
            x = alpha[None, :] * beta[:, None]
            s = _sigmoid(x)
            resid = (post_at_answer - s)     # (p_ij - sigma_ij)

            grad_alpha = (resid * beta[:, None]).sum(axis=0) - (alpha - mu_a) / sd_a**2
            grad_b = (resid * alpha[None, :]).sum(axis=1) * beta - (b - mu_b) / sd_b**2
            alpha += lr * grad_alpha
            b += lr * grad_b

        if abs(ll - prev_ll) < tol * (1 + abs(prev_ll)):
            break
        prev_ll = ll

    return pred, post_at_answer, alpha, np.exp(b)
