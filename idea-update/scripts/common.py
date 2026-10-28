"""Shared helpers for the idea-update round-2 checks (read-only on existing caches)."""
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import audit_revision_v1_9 as ar  # noqa: E402

OUT = ROOT / 'idea-update/results'
TARGETS = ('AllErr', 'FN')
BUDGET = .10
YEARS = (2021, 2022, 2023)
INSTS = ('freddie_mac', 'fannie_mae')


def lg(x):
    return logit(np.clip(x, 1e-6, 1 - 1e-6))


def logloss(q, y):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


class Learner:
    """additive / interaction follow src/audit_revision_v1_9.py; hgb_large and linear are pre-declared extras."""

    def __init__(self, kind, categorical=None):
        self.kind, self.categorical = kind, categorical

    def fit(self, x, y):
        if self.kind == 'additive':
            self.spline = SplineTransformer(n_knots=5, degree=3, knots='quantile', include_bias=False)
            self.scaler = StandardScaler()
            x = self.scaler.fit_transform(self.spline.fit_transform(x))
            self.model = LogisticRegression(C=1., solver='lbfgs', max_iter=4000, tol=1e-7)
        elif self.kind == 'linear':
            self.scaler = StandardScaler(); x = self.scaler.fit_transform(x)
            self.model = LogisticRegression(C=1., solver='lbfgs', max_iter=4000, tol=1e-7)
        elif self.kind in ('interaction', 'hgb_large'):
            big = self.kind == 'hgb_large'
            self.model = HistGradientBoostingClassifier(
                max_iter=300 if big else 100, max_leaf_nodes=15 if big else 7,
                min_samples_leaf=50 if big else 100, learning_rate=.05, l2_regularization=1.,
                max_bins=128 if not big else 255, early_stopping=False, random_state=ar.SEED,
                categorical_features=self.categorical)
        else:
            raise ValueError(self.kind)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            self.model.fit(x, y)
        return self

    def predict(self, x):
        if self.kind == 'additive':
            x = self.scaler.transform(self.spline.transform(x))
        elif self.kind == 'linear':
            x = self.scaler.transform(x)
        return self.model.predict_proba(x)[:, 1]


def cohort(inst, year):
    return ar.load_npz(ar.OLD / f'{inst}_{year}.npz')


def global_family_masks(inst):
    members, masks = ar.candidate_variants(f'{inst}_2018')
    return members, masks


def boot_hits(orders, zs, n, b, draws, seed, chunk=400):
    """Paired multinomial row-weight bootstrap (as in the paper): fixed rankings fill B weighted occurrences."""
    rng = np.random.default_rng(seed)
    hits = {k: np.empty(draws) for k in orders}
    totals = {t: np.empty(draws) for t in {k[1] for k in orders}}
    done = 0
    while done < draws:
        m = min(chunk, draws - done)
        w = rng.multinomial(n, np.full(n, 1 / n), size=m).astype(float)
        for t in totals:
            totals[t][done:done + m] = w @ zs[t]
        for k, o in orders.items():
            wo = w[:, o]
            sel = np.clip(b - (np.cumsum(wo, 1) - wo), 0, wo)
            hits[k][done:done + m] = sel @ zs[k[1]][o]
        done += m
    return hits, totals


def compare(z, tau, cand, q_by_policy, pairs, draws, seed, alpha=.05, family_size=1):
    """Target-aligned queues from fitted outcome probabilities; paired contrasts with bootstrap intervals.

    q_by_policy: policy -> fitted P(Y=1); pairs: list of (new, ref). Returns list of dict rows.
    """
    p, ids = z['p_dep'], z['row_id']
    n = len(ids); b = math.ceil(BUDGET * n)
    tv = ar.target_vectors(z, tau, cand)
    orders, zs = {}, {t: tv[t].astype(float) for t in TARGETS}
    for pol, q in q_by_policy.items():
        for t in TARGETS:
            orders[(pol, t)] = ar.order(ar.target_probability(q, p, tau, t), ids)
    bh, bt = boot_hits(orders, zs, n, b, draws, seed)
    a = alpha / family_size
    rows = []
    for new, ref in pairs:
        for t in TARGETS:
            T = int(tv[t].sum())
            h_new = int(tv[t][orders[(new, t)][:b]].sum()); h_ref = int(tv[t][orders[(ref, t)][:b]].sum())
            diff = (bh[(new, t)] - bh[(ref, t)])
            diff = diff[np.isfinite(diff)]
            ceiling = min(b, T) - h_ref
            lo, hi = np.quantile(diff, a / 2), np.quantile(diff, 1 - a / 2)  # in resampled hits
            rows.append(dict(new=new, ref=ref, target=t, N=n, B=b, T=T, hits_ref=h_ref, hits_new=h_new,
                             d_hits=h_new - h_ref, ceiling_ref=ceiling, lo_hits=float(lo), hi_hits=float(hi),
                             meaningful_gain=bool((h_new - h_ref) >= max(5, .10 * ceiling) and lo > 0),
                             excluded=bool(ceiling >= 50 and hi < .10 * ceiling),
                             interval=f'{"bonferroni" + str(family_size) if family_size > 1 else "pointwise95"}',
                             draws=len(diff)))
    return rows, orders
