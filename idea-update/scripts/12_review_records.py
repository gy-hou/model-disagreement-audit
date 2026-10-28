"""Saves the complete records behind the 10-22 review checks (no new estimation).

(a) Step 2 HGB (interaction) ID-R per cell, from results/02_contrasts.csv;
(b) share of later-cohort rows whose categorical level was unseen in 2018 (encoding check for B);
(c) same-seed G2-G1 hits from the v1.9 candidate reconstruction (results/tables/v1_9_candidate_rebuild/comparisons.csv).
"""
import joblib, numpy as np, pandas as pd
from common import ROOT, OUT, INSTS, YEARS

c = pd.read_csv(OUT / '02_contrasts.csv')
h = c[(c.learner == 'interaction') & (c.contrast == 'ID-R')].copy()
h['lo_hits'] = h.lo_pp / 100 * h['T']; h['hi_hits'] = h.hi_pp / 100 * h['T']
h['meaningful_gain'] = (h.d_hits >= np.maximum(5, .1 * h.ceiling_ref)) & (h.lo_pp > 0)
h['excluded'] = (h.ceiling_ref >= 50) & (h.hi_hits < .1 * h.ceiling_ref)
h[['inst', 'variant', 'year', 'target', 'T', 'B', 'ceiling_ref', 'hits_ref', 'hits_new', 'd_hits', 'lo_hits', 'hi_hits',
   'interval', 'draws', 'meaningful_gain', 'excluded']].to_csv(OUT / '12_hgb_id_minus_r.csv', index=False)

rows = []
for inst in INSTS:
    art = joblib.load(ROOT / f'models/v0_7/deployment_grid/{inst}_2018/candidates/M2_xgb_d3_lr03_n400.joblib')
    f18 = pd.read_parquet(ROOT / f'data/processed/{inst}_2018_ever_60dpd_24m/test.parquet')
    for yr in YEARS:
        f = pd.read_parquet(ROOT / f'data/processed/{inst}_{yr}_ever_60dpd_24m/test.parquet')
        for col in art['categorical_features']:
            lv = set(f18[col].dropna().astype(str))
            rows.append(dict(inst=inst, year=yr, feature=col, levels_2018=len(lv), native_categorical=len(lv) <= 120,
                             share_unseen=float((~f[col].astype(str).isin(lv)).mean())))
pd.DataFrame(rows).to_csv(OUT / '12_unseen_category_share.csv', index=False)

cmp = pd.read_csv(ROOT / 'results/tables/v1_9_candidate_rebuild/comparisons.csv')
cmp[['base', 'cohort', 'seed', 'M', 'target', 'G1_hits', 'G2_hits', 'hits_difference', 'queue_jaccard', 'lower', 'upper']] \
    .to_csv(OUT / '12_rebuild_same_seed_hits.csv', index=False)
print('saved 12_hgb_id_minus_r.csv, 12_unseen_category_share.csv, 12_rebuild_same_seed_hits.csv')
u = pd.DataFrame(rows); print(u[u.share_unseen > 0].round(4).to_string(index=False))
r = pd.read_csv(OUT / '12_rebuild_same_seed_hits.csv')
print(r[r.cohort.str.contains('2021|2022|2023')].pivot_table(index=['cohort', 'target'], columns='seed', values='hits_difference').to_string())
