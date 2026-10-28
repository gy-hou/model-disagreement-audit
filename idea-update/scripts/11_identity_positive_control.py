"""Positive control (GPT Pro's suggestion): can Step 2's learners recover a KNOWN identity gain?

Data from the 9-cell construction: P1, P2 iid uniform on {pi-d, pi, pi+d}; eta = p_i + p_j - pi + eps*A_ij;
S = calibrated two-level score (M >= pi vs M < pi); target AllErr at 10%. R = [logit S, logit C, D] (= SYM here);
ID = R + logit(P1), logit(P2). Learners use Step 2's exact settings. Gains are conditional expected hits from true eta.
"""
import numpy as np, pandas as pd, importlib.util, pathlib
from common import Learner, lg, OUT
spec = importlib.util.spec_from_file_location('ex', pathlib.Path(__file__).with_name('00_exact_construction.py'))
ex = importlib.util.module_from_spec(spec)
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(ex)
from fractions import Fraction as F

def setting(pi, d, eps):
    cells, _ = ex.build(F(pi), F(d), F(eps))
    P = np.array([[float(c['p1']), float(c['p2'])] for c in cells]); eta = np.array([float(c['eta']) for c in cells])
    S = np.array([float(c['S']) for c in cells]); yhat = np.array([bool(c['yhat']) for c in cells])
    err = np.where(yhat, 1 - eta, eta)
    oracle = float(ex.oracle_value(cells, lambda c: (c['S'], c['p1'], c['p2']), F(1, 10)) - ex.oracle_value(cells, lambda c: (c['S'], tuple(sorted((c['p1'], c['p2'])))), F(1, 10)))
    return P, eta, S, yhat, err, oracle

def feats(P, S, rep):
    cols = [lg(S), lg(P.mean(1)), np.abs(P[:, 0] - P[:, 1])]
    if rep == 'ID': cols += [lg(P[:, 0]), lg(P[:, 1])]
    return np.column_stack(cols)

rows = []
SETTINGS = [('pi.5_eps.1', .5, .1, .1), ('pi.5_eps.05', .5, .1, .05), ('pi.04_eps.01', .04, .01, .01)]
SETTING_ID = {'pi.5_eps.1': 1, 'pi.5_eps.05': 2, 'pi.04_eps.01': 3}
for name, pi, d, eps in SETTINGS:
    P9, eta9, S9, yhat9, err9, oracle = setting(pi, d, eps)
    for n_dev in (10_000, 80_000):
        for r in range(100):
            rng = np.random.default_rng([20261022, SETTING_ID[name], n_dev, r])   # fixed ids (v1.11 fix: str hash() is salted per process)
            cell_dev = rng.integers(0, 9, n_dev); y_dev = (rng.random(n_dev) < eta9[cell_dev]).astype(int)
            cell_ev = rng.integers(0, 9, 10_000); tie = rng.permutation(10_000)
            B = 1000; exp_err_total = err9[cell_ev].sum()
            for learner in ('interaction', 'additive'):
                gains = {}
                for rep in ('R', 'ID'):
                    m = Learner(learner).fit(feats(P9[cell_dev], S9[cell_dev], rep), y_dev)
                    q = m.predict(feats(P9[cell_ev], S9[cell_ev], rep))
                    t = np.where(yhat9[cell_ev], 1 - q, q)                       # target-aligned AllErr probability
                    sel = np.lexsort((tie, -np.round(t, 12)))[:B]
                    gains[rep] = err9[cell_ev][sel].sum()                       # E[hits | queue]
                rows.append(dict(setting=name, n_dev=n_dev, rep=r, learner=learner, oracle_hits_per10k=10000 * oracle,
                                 exp_gain_hits=gains['ID'] - gains['R'], exp_err_total=exp_err_total))
    print('done', name, flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / '11_identity_positive_control.csv', index=False)
g = df.groupby(['setting', 'n_dev', 'learner']).agg(oracle=('oracle_hits_per10k', 'first'), mean_gain=('exp_gain_hits', 'mean'),
        sd=('exp_gain_hits', 'std'), share_pos=('exp_gain_hits', lambda x: (x > 0).mean()))
g['mc_lo'] = g.mean_gain - 1.96 * g.sd / 10; g['mc_hi'] = g.mean_gain + 1.96 * g.sd / 10; g['recovered_share'] = g.mean_gain / g.oracle
pd.set_option('display.width', 200); print(g.round(3).to_string())
