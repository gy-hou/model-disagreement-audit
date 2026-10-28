"""Branch F: construction E with audit learners REFITTED on the candidates that actually pass a finite validation gate.

Independent re-implementation (new seeds, not v1.10's streams). Saturated Beta(1/2,1/2) learner; Pos target at 10%.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import OUT

mass_g = np.array([.6, .2, .1, .1]); S_g = np.array([.04, .08, .11, .112])
S = np.repeat(S_g, 2); U = np.tile([-1., 1.], 4); mass = np.repeat(mass_g / 2, 2)
q = S + .0025 * U; D = .0004 + .0002 * U
CAND = np.column_stack([S - D / 2, S, S + D / 2])           # lower, center, upper (per atom)

def sample(n, rng):
    counts = np.rint(mass * n).astype(int); atom = np.repeat(np.arange(8), counts)
    return atom, (rng.random(len(atom)) < q[atom]).astype(int)

def sat_rates(keys_atom, atom_tr, y_tr):
    """Saturated estimate per unique key: (k + .5)/(n + 1); returns rate per atom."""
    _, inv = np.unique(np.round(keys_atom, 12), axis=0, return_inverse=True); inv = inv.ravel()   # round: float noise must not leak U
    g_tr = inv[atom_tr]; k = np.bincount(g_tr, weights=y_tr, minlength=inv.max() + 1); n = np.bincount(g_tr, minlength=inv.max() + 1)
    return ((k + .5) / (n + 1))[inv]

def oracle_gain_pp(keys1, keys2, b=.10):
    """Population value: rank groups by true mass-weighted risk, fill mass b; gain in capture pp."""
    def value(keys):
        _, inv = np.unique(np.round(keys, 12), axis=0, return_inverse=True); inv = inv.ravel()
        gm = np.bincount(inv, weights=mass); gr = np.bincount(inv, weights=mass * q) / gm
        left, v = b, 0.
        for g in np.argsort(-gr):
            take = min(left, gm[g]); v += take * gr[g]; left -= take
            if left <= 1e-15: break
        return v
    return 100 * (value(keys2) - value(keys1)) / (mass * q).sum()

def capture(rate_atom, atom_ev, y_ev, rng, b=.10):
    """Realized capture, plus E[hits | queue] computed from the true risks q (removes evaluation-label noise)."""
    B = int(np.ceil(b * len(atom_ev))); tie = rng.permutation(len(atom_ev))
    order = np.lexsort((tie, -np.round(rate_atom[atom_ev], 12))); sel = order[:B]
    return y_ev[sel].sum() / y_ev.sum(), q[atom_ev[sel]].sum()

rows = []
for s in range(300):
    rng = np.random.default_rng([20261021, s])
    a_val, y_val = sample(10_000, rng)
    auc = np.array([roc_auc_score(y_val, CAND[a_val, j]) for j in range(3)])
    keep = auc + 1e-6 >= auc.max() - .010
    C_ret = CAND[:, keep].mean(1); D_ret = CAND[:, keep].max(1) - CAND[:, keep].min(1)
    a_ev, y_ev = sample(10_000, rng)
    for n_tr in (8_000, 80_000):
        a_tr, y_tr = sample(n_tr, rng)
        res = {}
        for label, C_, D_ in (('fixed3', S, D), ('postgate', C_ret, D_ret)):
            r1 = sat_rates(np.column_stack([S, C_]), a_tr, y_tr); r2 = sat_rates(np.column_stack([S, C_, D_]), a_tr, y_tr)
            ev_rng = np.random.default_rng([20261021, s, n_tr, 7])
            (c1, e1) = capture(r1, a_ev, y_ev, ev_rng); ev_rng = np.random.default_rng([20261021, s, n_tr, 7]); (c2, e2) = capture(r2, a_ev, y_ev, ev_rng)
            exp_T = q[a_ev].sum()
            og = oracle_gain_pp(np.column_stack([S, C_]), np.column_stack([S, C_, D_]))
            rows.append(dict(seed=s, n_train=n_tr, set=label, M_ret=int(keep.sum()), kept=''.join('LCU'[j] for j in range(3) if keep[j]),
                             cap_G1=c1, cap_G2=c2, gain_pp=100 * (c2 - c1), oracle_gain_pp=og,
                             exp_gain_pp=100 * (e2 - e1) / exp_T))
df = pd.DataFrame(rows); df.to_csv(OUT / '09_postgate_refit_sim.csv', index=False)
def summ(g):
    m, sd, n = g.gain_pp.mean(), g.gain_pp.std(ddof=1), len(g)
    return pd.Series(dict(n=n, mean_pp=m, mc_lo=m - 1.96 * sd / np.sqrt(n), mc_hi=m + 1.96 * sd / np.sqrt(n),
                          q025=g.gain_pp.quantile(.025), q975=g.gain_pp.quantile(.975), mean_cap_G1=100 * g.cap_G1.mean(),
                          oracle_mean_pp=g.oracle_gain_pp.mean(), exp_mean_pp=g.exp_gain_pp.mean(),
                          exp_mc_lo=g.exp_gain_pp.mean() - 1.96 * g.exp_gain_pp.std(ddof=1) / np.sqrt(n),
                          exp_mc_hi=g.exp_gain_pp.mean() + 1.96 * g.exp_gain_pp.std(ddof=1) / np.sqrt(n)))
pd.set_option('display.width', 200)
print(df.groupby(['set', 'n_train']).apply(summ, include_groups=False).round(3).to_string())
print(); print(df[df.set == 'postgate'].groupby(['n_train', 'M_ret', 'kept']).apply(summ, include_groups=False).round(3).to_string())
