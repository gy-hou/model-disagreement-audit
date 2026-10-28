"""POST-HOC robustness (added after seeing Step 2): low-variance linear stacking learner.

Same representations, data, budget, targets and bootstrap as 02; learner = standardized
features + L2 logistic regression (C=1), i.e. classic linear stacking on candidate logits.
"""
import importlib.util, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rc', HERE / '02_representation_compare.py')
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
ar = rc.ar


class Linear:
    def __init__(self, rep): self.rep = rep
    def fit(self, p, c, y):
        x = rc.features(p, c, self.rep); self.sc = StandardScaler()
        self.m = LogisticRegression(C=1., solver='lbfgs', max_iter=4000, tol=1e-7).fit(self.sc.fit_transform(x), y)
        return self
    def predict(self, p, c): return self.m.predict_proba(self.sc.transform(rc.features(p, c, self.rep)))[:, 1]


rows, dev = [], []
for inst in ('freddie_mac', 'fannie_mae'):
    base = f'{inst}_2018'; tau = ar.TAUS[base]; _, masks = ar.candidate_variants(base)
    d = ar.load_npz(ar.OLD / f'{base}.npz')
    folds = list(StratifiedKFold(5, shuffle=True, random_state=ar.SEED).split(d['p_dep'], d['y']))
    for variant in ('global', 'family'):
        mk = masks[variant]; cd = d['cand'][:, mk]
        models = {r: Linear(r).fit(d['p_dep'], cd, d['y']) for r in rc.REPS}
        oof = {r: np.empty(len(d['y'])) for r in rc.REPS}
        for tr, va in folds:
            for r in rc.REPS:
                oof[r][va] = Linear(r).fit(d['p_dep'][tr], cd[tr], d['y'][tr]).predict(d['p_dep'][va], cd[va])
        tvd = ar.target_vectors(d, tau, cd)
        for r in rc.REPS:
            rec = dict(inst=inst, variant=variant, rep=r, oof_logloss=rc.logloss(oof[r], d['y']))
            for t in rc.TARGETS:
                rec[f'oof_hits_{t}'] = int(tvd[t][ar.queue(ar.target_probability(oof[r], d['p_dep'], tau, t), d['row_id'], .10)].sum())
            dev.append(rec)
        for yr in (2021, 2022, 2023):
            z = ar.load_npz(ar.OLD / f'{inst}_{yr}.npz'); p, c, ids, y = z['p_dep'], z['cand'][:, mk], z['row_id'], z['y']
            n = len(ids); b = math.ceil(.10 * n); tv = ar.target_vectors(z, tau, c)
            orders, zs = {}, {t: tv[t].astype(float) for t in rc.TARGETS}; ll = {}
            for r in rc.REPS:
                q = models[r].predict(p, c); ll[r] = rc.logloss(q, y)
                for t in rc.TARGETS:
                    orders[(r, t)] = ar.order(ar.target_probability(q, p, tau, t), ids)
            bh, bt = rc.boot_hits(orders, zs, n, b, 1999, ar.SEED + 1000 * yr + (11 if inst == 'freddie_mac' else 13))
            for t in rc.TARGETS:
                h = {r: int(tv[t][orders[(r, t)][:b]].sum()) for r in rc.REPS}; T = int(tv[t].sum())
                cap = (bh[('ID', t)] - bh[('R', t)]) / bt[t]; cap = cap[np.isfinite(cap)]
                rows.append(dict(inst=inst, year=yr, variant=variant, target=t, T=T, B=b, hits_R=h['R'], hits_SYM=h['SYM'],
                                 hits_ID=h['ID'], d_hits_ID_R=h['ID'] - h['R'], ceiling_R=min(b, T) - h['R'],
                                 lo95_pp=100 * np.quantile(cap, .025), hi95_pp=100 * np.quantile(cap, .975),
                                 logloss_R=ll['R'], logloss_ID=ll['ID'], d_logloss_ID_R=ll['ID'] - ll['R']))
pd.DataFrame(rows).to_csv(rc.OUT / '03_posthoc_linear_stacking.csv', index=False)
pd.DataFrame(dev).to_csv(rc.OUT / '03_posthoc_linear_stacking_oof.csv', index=False)
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).round(5).to_string(index=False)); print(); print(pd.DataFrame(dev).round(5).to_string(index=False))
