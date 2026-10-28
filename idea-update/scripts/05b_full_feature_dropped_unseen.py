"""Branch B (05b variant: POST-HOC, drops fields whose later-cohort levels are mostly unseen): feasible headroom. G_X = [logit S, logit C, all origination features X] vs G1, HGB learners.

Candidates are deterministic functions of X, so I(Y; any candidate summary | S, C) <= I(Y; X | S, C):
Positioning (10-22): a full-feature INCREMENTAL COMPARISON of fitted policies, not an oracle bound; a null here
does not show that optimal remaining value or conditional information is zero.
Encoding is fixed on 2018 C_audit; unseen levels -> missing; >120 levels -> numeric (zip prefixes parsed).
"""
import joblib, numpy as np, pandas as pd
from common import ar, Learner, lg, cohort, global_family_masks, compare, logloss, OUT, INSTS, YEARS, ROOT

def load_frame(inst, yr, row_id):
    f = pd.read_parquet(ROOT / f'data/processed/{inst}_{yr}_ever_60dpd_24m/test.parquet').set_index('row_id').loc[row_id]
    return f

class Encoder:
    def __init__(self, raw, cat): self.raw, self.cat = raw, cat
    def fit(self, f):
        self.levels, self.native = {}, []
        for c in self.cat:
            lv = sorted(f[c].dropna().astype(str).unique())
            if len(lv) <= 120: self.levels[c] = {v: i for i, v in enumerate(lv)}; self.native.append(c)
        return self
    def transform(self, f):
        cols = []
        for c in self.raw:
            if c in self.levels: cols.append(f[c].astype(str).map(self.levels[c]).astype(float).to_numpy())
            else: cols.append(pd.to_numeric(f[c], errors='coerce').to_numpy(dtype=float))
        return np.column_stack(cols)
    def mask(self, n_prefix): return np.array([False] * n_prefix + [c in self.levels for c in self.raw])

DROP = {'freddie_mac': ['property_valuation_method'], 'fannie_mae': ['origination_date', 'first_payment_date']}
rows = []
for inst in INSTS:
    base = f'{inst}_2018'; tau = ar.TAUS[base]; _, masks = global_family_masks(inst); mk = masks['global']
    art = joblib.load(ROOT / f'models/v0_7/deployment_grid/{base}/candidates/M2_xgb_d3_lr03_n400.joblib')
    d = cohort(inst, 2018); fd = load_frame(inst, 2018, d['row_id'])
    assert (fd['ever_60dpd_24m'].to_numpy() == d['y']).all()
    enc = Encoder([c for c in art['raw_feature_columns'] if c not in DROP[inst]], [c for c in art['categorical_features'] if c not in DROP[inst]]).fit(fd)
    def design(z, f, with_x):
        base_cols = np.column_stack([lg(z['p_dep']), lg(z['cand'][:, mk].mean(1))])
        return np.hstack([base_cols, enc.transform(f)]) if with_x else base_cols
    catmask = enc.mask(2)
    models = {
        'G1_add': Learner('additive').fit(design(d, fd, False), d['y']),
        'G1_int': Learner('interaction').fit(design(d, fd, False), d['y']),
        'G1_large': Learner('hgb_large').fit(design(d, fd, False), d['y']),
        'GX_int': Learner('interaction', categorical=catmask).fit(design(d, fd, True), d['y']),
        'GX_large': Learner('hgb_large', categorical=catmask).fit(design(d, fd, True), d['y']),
    }
    for yr in YEARS:
        z = cohort(inst, yr); f = load_frame(inst, yr, z['row_id'])
        assert (f['ever_60dpd_24m'].to_numpy() == z['y']).all()
        q = {k: m.predict(design(z, f, k.startswith('GX'))) for k, m in models.items()}
        rr, _ = compare(z, tau, z['cand'][:, mk], q, [('GX_int', 'G1_int'), ('GX_large', 'G1_large'), ('GX_large', 'G1_add')],
                        1999, ar.SEED + 53 * yr + len(inst))
        for r in rr:
            r.update(inst=inst, year=yr, logloss_new=logloss(q[r['new']], z['y']), logloss_ref=logloss(q[r['ref']], z['y']))
        rows += rr
df = pd.DataFrame(rows); df.to_csv(OUT / '05b_full_feature_dropped_unseen.csv', index=False)
pd.set_option('display.width', 250)
print(df[['inst', 'year', 'new', 'ref', 'target', 'T', 'hits_ref', 'hits_new', 'd_hits', 'ceiling_ref', 'lo_hits', 'hi_hits',
          'meaningful_gain', 'excluded', 'logloss_ref', 'logloss_new']].round(5).to_string(index=False))
