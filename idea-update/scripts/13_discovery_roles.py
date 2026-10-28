"""Data roles for the audit discovery evaluation (descriptive metadata only; no method comparisons).

For each of the 7 real discovery datasets: sizes and prevalences of the original splits, label definition,
threshold source, cached candidate set and global .010 subset, reference-model identity (exact match against
candidate columns), base-model training/selection split fields, ID-based feature alignment, redundancy flags.
"""
import json, sys
import joblib, numpy as np, pandas as pd
from common import ROOT, OUT, ar
sys.path.insert(0, str(ROOT))
from src.audit_budget_routing_v0_6 import DATASET_CONFIGS, find_deployed_m2, resolve, load_frame

DATASETS = ('uci', 'fico', 'hmda', 'lendingclub', 'gmc', 'hmeq', 'home_credit')
LABELS = {'uci': 'default payment next month (credit-card default)', 'fico': 'is_at_risk (HELOC RiskPerformance = Bad)',
          'hmda': 'y = application originated (NOT repayment)', 'lendingclub': 'default (charged off / default)',
          'gmc': 'SeriousDlqin2yrs (90+ days past due within 2 years)', 'hmeq': 'BAD (home-equity loan default or seriously delinquent)',
          'home_credit': 'TARGET (payment difficulties)'}
inv = pd.read_csv(ROOT / 'results/tables/v1_4_mechanism/cohort_inventory.csv').set_index('cohort')
rows = []
for name in DATASETS:
    cfg = DATASET_CONFIGS[name]
    z = np.load(ar.OLD / f'{name}.npz'); ids, y, p, cand = z['row_id'], z['y'], z['p_dep'], z['cand']
    mem = pd.read_csv(ar.OLD / f'{name}_members.csv')
    keep = mem.validation_auc.to_numpy() + 1e-6 >= mem.validation_auc.max() - .010
    test = load_frame(cfg.test_path); val = load_frame(cfg.validation_path)
    train_path = Path_ = str(cfg.test_path).replace('test', 'train')
    try: tr = load_frame(type(cfg.test_path)(train_path)); n_train = len(tr); prev_train = float(tr[cfg.target].mean())
    except Exception: n_train, prev_train = None, None
    tau_val = float(val[cfg.target].mean())
    voters = json.loads(resolve(cfg.voters_path).read_text())['selected_voters']; dep = find_deployed_m2(voters)
    diffs = np.abs(cand - p[:, None]).max(0); j = int(diffs.argmin())
    s_in_cache = bool(diffs[j] < 1e-12); s_id = mem.candidate_id.iloc[j] if s_in_cache else None
    art = joblib.load(resolve(type(cfg.test_path)(dep['artifact_path'])))
    feats = list(art.get('raw_feature_columns') or art.get('feature_columns'))
    f = test.set_index(cfg.id_column)
    aligned = set(ids) <= set(f.index) and bool((f.loc[ids, cfg.target].to_numpy() == y).all())
    missing_cols = [c for c in feats if c not in f.columns]
    Mg = int(keep.sum()); s_in_global = bool(s_in_cache and keep[j])
    if Mg < 2: flag = 'M_global<2: D=0, G2/SYM/ID degenerate (not defined)'
    elif Mg == 2 and s_in_global: flag = 'M_global=2 and S in set: SYM≡G2 algebraically; ID recoverable from G1 (S,C) -> ID-G1, ID-SYM cannot reflect new information'
    elif Mg == 2: flag = 'M_global=2: SYM≡G2 algebraically; ID adds only sign(P1-P2)'
    elif s_in_global: flag = f'M_global={Mg}, S in set: other {Mg-1} candidates not recoverable from (S,C)'
    else: flag = f'M_global={Mg}, S not in set'
    rows.append(dict(dataset=name, label=LABELS[name], y_semantic=cfg.y_semantic,
                     n_train=n_train, n_val=len(val), n_test=len(test), prev_train=prev_train,
                     prev_val=tau_val, prev_test=float(y.mean()),
                     tau=float(inv.loc[name, 'tau']), tau_source='validation prevalence',
                     tau_matches_val=bool(abs(inv.loc[name, 'tau'] - tau_val) < 1e-12),
                     M_cached_family=int(cand.shape[1]), M_global=Mg,
                     global_families=','.join(sorted(mem.model_id[keep].astype(str).unique())),
                     reference_model=dep['candidate_id'], reference_in_cache=s_in_cache, reference_cache_id=s_id,
                     reference_in_global=s_in_global, redundancy=flag,
                     ref_trained_on=str(art.get('trained_on')), ref_selection_split=str(art.get('selection_split')),
                     id_column=cfg.id_column, ids_unique=bool(len(np.unique(ids)) == len(ids)),
                     features_aligned_by_id=aligned, n_raw_features=len(feats), missing_feature_columns=';'.join(missing_cols),
                     categorical_features=';'.join(cfg.categorical_features)))
df = pd.DataFrame(rows); df.to_csv(OUT / '13_discovery_roles.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
print(df.T.to_string())
