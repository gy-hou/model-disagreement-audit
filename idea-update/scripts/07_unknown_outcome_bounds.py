"""Branch D: identification bounds for frozen-queue hit differences under the strict calendar endpoint (Freddie).

Delta H in [d - u_minus, d + u_plus]: d = known-target hit difference on the symmetric difference of the queues;
u_plus / u_minus = unknown-target cases only in the new / only in the old queue. Shared members cancel.
"""
import math, numpy as np, pandas as pd
from common import ar, Learner, lg, cohort, global_family_masks, OUT, YEARS, ROOT

inst = 'freddie_mac'; tau = ar.TAUS[f'{inst}_2018']; _, masks = global_family_masks(inst); mk = masks['global']
d = cohort(inst, 2018)
def feats(p, c, lvl):
    cols = [lg(p), lg(c.mean(1))] + ([np.ptp(c, 1)] if lvl == 2 else [])
    return np.column_stack(cols)
G = {k: Learner('additive').fit(feats(d['p_dep'], d['cand'][:, mk], lvl), d['y']) for k, lvl in (('G1', 1), ('G2', 2))}
rows = []
for yr in YEARS:
    z = cohort(inst, yr); p, ids, c = z['p_dep'], z['row_id'], z['cand'][:, mk]
    n = len(ids); b = math.ceil(.10 * n); pred = p >= tau
    a = pd.read_parquet(ROOT / f'results/tables/v1_9_revision/local_cache/{inst}_{yr}_strict_label_audit.parquet').set_index('row_id').loc[ids]
    sy = a['strict_y'].to_numpy()
    strict = {'AllErr': np.where(sy < 0, -1, (pred != (sy == 1)).astype(int)),
              'FN': np.where(pred, 0, np.where(sy < 0, -1, sy))}
    legacy = ar.target_vectors(z, tau, c)
    q = {k: m.predict(feats(p, c, 1 if k == 'G1' else 2)) for k, m in G.items()}
    queues = {}
    for t in ('AllErr', 'FN'):
        for k in ('G1', 'G2'):
            queues[(k, t)] = set(ar.queue(ar.target_probability(q[k], p, tau, t), ids, .10))
    queues[('AboveTauLow', 'AllErr')] = set(ar.order(np.where(pred, -p, p - 2), ids)[:b])
    for new, old, t in (('G2', 'G1', 'AllErr'), ('G2', 'G1', 'FN'), ('G2', 'AboveTauLow', 'AllErr')):
        A = np.fromiter(queues[(new, t)] - queues[(old, t)], int); Bs = np.fromiter(queues[(old, t)] - queues[(new, t)], int)
        zt = strict[t]
        dk = int((zt[A] == 1).sum() - (zt[Bs] == 1).sum()); up = int((zt[A] < 0).sum()); um = int((zt[Bs] < 0).sum())
        Tl = int(legacy[t].sum()); h_old_leg = int(legacy[t][np.fromiter(queues[(old, t)], int)].sum())
        h_new_leg = int(legacy[t][np.fromiter(queues[(new, t)], int)].sum()); ceil = min(b, Tl) - h_old_leg
        thr = max(5, .10 * ceil)
        rows.append(dict(year=yr, contrast=f'{new}-{old}', target=t, exchanged=len(A), legacy_dH=h_new_leg - h_old_leg,
                         known_d=dk, u_plus=up, u_minus=um, bound_lo=dk - um, bound_hi=dk + up, legacy_ceiling=ceil,
                         materiality=thr, cannot_create_meaningful_gain=bool(dk + up < thr),
                         cannot_create_meaningful_loss=bool(dk - um > -thr),
                         strict_unknown_in_cohort=int((zt < 0).sum())))
df = pd.DataFrame(rows); df.to_csv(OUT / '07_unknown_outcome_bounds.csv', index=False)
pd.set_option('display.width', 250); print(df.to_string(index=False))
