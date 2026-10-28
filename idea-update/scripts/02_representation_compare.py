"""Step 2 of the identity-information branch: R = (S, C, D) vs SYM (+ sorted P) vs ID (+ ordered P).

Read-only on existing caches. Learners, folds, budgets, targets and the bootstrap scheme follow
src/audit_revision_v1_9.py; only the audit-model representation changes.
Pre-declared plan and decision rules: idea-update/IDEA_UPDATE.md (Step 2).
"""
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import audit_revision_v1_9 as ar  # noqa: E402

OUT = ROOT / 'idea-update/results'
REPS = ('R', 'SYM', 'ID')
TARGETS = ('AllErr', 'FN')
BUDGET = .10
CONTRASTS = (('ID', 'R'), ('SYM', 'R'), ('ID', 'SYM'))
PRIMARY = dict(variant='global', learner='additive', contrast='ID-R')


def lg(x):
    return logit(np.clip(x, 1e-6, 1 - 1e-6))


def features(p, cand, rep):
    cols = [lg(p)[:, None], lg(cand.mean(1))[:, None], np.ptp(cand, 1)[:, None]]
    if rep in ('SYM', 'ID'):
        cols.append(lg(np.sort(cand, 1)))
    if rep == 'ID':
        cols.append(lg(cand))
    return np.hstack(cols)


class Model:
    def __init__(self, rep, learner):
        self.rep, self.learner = rep, learner

    def fit(self, p, cand, y):
        x = features(p, cand, self.rep)
        if self.learner == 'additive':
            self.spline = SplineTransformer(n_knots=5, degree=3, knots='quantile', include_bias=False)
            self.scaler = StandardScaler()
            x = self.scaler.fit_transform(self.spline.fit_transform(x))
            self.model = LogisticRegression(C=1., solver='lbfgs', max_iter=4000, tol=1e-7)
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=100, max_leaf_nodes=7, min_samples_leaf=100, learning_rate=.05,
                l2_regularization=1., max_bins=128, early_stopping=False, random_state=ar.SEED)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            self.model.fit(x, y)
        return self

    def predict(self, p, cand):
        x = features(p, cand, self.rep)
        if self.learner == 'additive':
            x = self.scaler.transform(self.spline.transform(x))
        return self.model.predict_proba(x)[:, 1]


def logloss(q, y):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def boot_hits(orders, zs, n, b, draws, seed, chunk=400):
    """Paired multinomial row-weight bootstrap; fixed rankings fill B weighted occurrences.

    orders: dict key -> index order; zs: dict key -> target indicator (float); returns
    dict key -> (draws,) resampled hits, and dict target -> (draws,) resampled target totals.
    """
    rng = np.random.default_rng(seed)
    hits = {k: np.empty(draws) for k in orders}
    totals = {}
    done = 0
    while done < draws:
        m = min(chunk, draws - done)
        w = rng.multinomial(n, np.full(n, 1 / n), size=m).astype(float)
        for t in {k[1] for k in orders}:
            totals.setdefault(t, np.empty(draws))[done:done + m] = w @ zs[t]
        for k, o in orders.items():
            wo = w[:, o]
            prior = np.cumsum(wo, 1) - wo
            sel = np.clip(b - prior, 0, wo)
            hits[k][done:done + m] = sel @ zs[k[1]][o]
        done += m
    return hits, totals


def main():
    rows, contrasts, dev = [], [], []
    for inst in ('freddie_mac', 'fannie_mae'):
        base = f'{inst}_2018'
        tau = ar.TAUS[base]
        _, masks = ar.candidate_variants(base)
        d = ar.load_npz(ar.OLD / f'{base}.npz')
        folds = list(StratifiedKFold(5, shuffle=True, random_state=ar.SEED).split(d['p_dep'], d['y']))
        for variant in ('global', 'family'):
            mk = masks[variant]
            cd = d['cand'][:, mk]
            for learner in ('additive', 'interaction'):
                models = {r: Model(r, learner).fit(d['p_dep'], cd, d['y']) for r in REPS}
                # development OOF diagnostic on C_audit
                oof = {r: np.empty(len(d['y'])) for r in REPS}
                for tr, va in folds:
                    for r in REPS:
                        oof[r][va] = Model(r, learner).fit(d['p_dep'][tr], cd[tr], d['y'][tr]).predict(d['p_dep'][va], cd[va])
                tv = ar.target_vectors(d, tau, cd)
                for r in REPS:
                    rec = dict(inst=inst, variant=variant, learner=learner, rep=r, M=int(mk.sum()),
                               oof_logloss=logloss(oof[r], d['y']))
                    for t in TARGETS:
                        q = ar.queue(ar.target_probability(oof[r], d['p_dep'], tau, t), d['row_id'], BUDGET)
                        rec[f'oof_hits_{t}'] = int(tv[t][q].sum())
                    dev.append(rec)
                for yr in (2021, 2022, 2023):
                    z = ar.load_npz(ar.OLD / f'{inst}_{yr}.npz')
                    p, c, ids, y = z['p_dep'], z['cand'][:, mk], z['row_id'], z['y']
                    n = len(ids); b = math.ceil(BUDGET * n)
                    tv = ar.target_vectors(z, tau, c)
                    orders, zs = {}, {t: tv[t].astype(float) for t in TARGETS}
                    for r in REPS:
                        q = models[r].predict(p, c)
                        for t in TARGETS:
                            o = ar.order(ar.target_probability(q, p, tau, t), ids)
                            orders[(r, t)] = o
                            h = int(tv[t][o[:b]].sum())
                            rows.append(dict(inst=inst, year=yr, variant=variant, learner=learner, rep=r,
                                             target=t, N=n, B=b, T=int(tv[t].sum()), hits=h,
                                             logloss=logloss(q, y)))
                    primary_cfg = variant == PRIMARY['variant'] and learner == PRIMARY['learner']
                    draws = 9999 if primary_cfg else 1999
                    seed = ar.SEED + 1000 * yr + (0 if inst == 'freddie_mac' else 7)
                    bh, bt = boot_hits(orders, zs, n, b, draws, seed)
                    for a, r0 in CONTRASTS:
                        for t in TARGETS:
                            cap = (bh[(a, t)] - bh[(r0, t)]) / np.where(bt[t] > 0, bt[t], np.nan)
                            cap = cap[np.isfinite(cap)]
                            obs_a = int(tv[t][orders[(a, t)][:b]].sum()); obs_0 = int(tv[t][orders[(r0, t)][:b]].sum())
                            T = int(tv[t].sum())
                            is_primary = primary_cfg and f'{a}-{r0}' == PRIMARY['contrast']
                            alpha = .05 / 12 if is_primary else .05
                            contrasts.append(dict(
                                inst=inst, year=yr, variant=variant, learner=learner, contrast=f'{a}-{r0}',
                                target=t, T=T, B=b, hits_ref=obs_0, hits_new=obs_a, d_hits=obs_a - obs_0,
                                ceiling_ref=min(b, T) - obs_0, d_capture_pp=100 * (obs_a - obs_0) / T,
                                lo_pp=100 * np.quantile(cap, alpha / 2), hi_pp=100 * np.quantile(cap, 1 - alpha / 2),
                                interval='bonferroni12' if is_primary else 'pointwise95', draws=len(cap)))
                    print(f'done {inst} {variant} {learner} {yr}', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / '02_hits.csv', index=False)
    pd.DataFrame(contrasts).to_csv(OUT / '02_contrasts.csv', index=False)
    pd.DataFrame(dev).to_csv(OUT / '02_development_oof.csv', index=False)


if __name__ == '__main__':
    main()
