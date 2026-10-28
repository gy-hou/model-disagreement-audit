"""Rebuild manuscript aggregates from verified v1.9 execution receipts.

This step performs no fitting, policy selection, or new statistical evaluation.
Run from any directory with the repository .venv Python.
"""
from pathlib import Path
import datetime
import hashlib
import json
import sys
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from audit_revision_v1_9 import check_development


def collect(scope, name):
    return pd.concat([pd.read_csv(p) for p in sorted((OUT/scope).glob(f'*/*/{name}.csv'))], ignore_index=True)


def records(frame):
    return json.loads(frame.to_json(orient='records', double_precision=15))


def main():
    check_development()
    h=collect('exploratory_retrospective','paired_contrasts')
    perf=collect('exploratory_retrospective','performance')
    pred=collect('exploratory_retrospective','predictive')
    swap=collect('exploratory_retrospective','budget_boundary')
    dec=collect('exploratory_retrospective','decomposition')
    main=h[(h.variant=='global')&(h.learner=='additive')&h.target.isin(['AllErr','FN'])&(h.budget==.1)&h.contrast.isin(['G2-G1','P_full-P_base'])].sort_values(['cohort','target','contrast'])
    sensitivity=h[h.target.isin(['AllErr','FN'])&(h.budget==.1)&(h.contrast=='G2-G1')].sort_values(['cohort','target','variant','learner'])
    mainperf=perf[(perf.variant=='global')&(perf.learner=='additive')&perf.target.isin(['AllErr','FN'])&(perf.budget==.1)]
    mainpred=pred[(pred.variant=='global')&(pred.learner=='additive')]
    mainswap=swap[(swap.variant=='global')&(swap.learner=='additive')&(swap.budget==.1)]
    s=collect('exploratory_strict_endpoint_eligible','paired_contrasts')
    sp=collect('exploratory_strict_endpoint_eligible','performance')
    strict=s[(s.variant=='global')&(s.learner=='additive')&s.target.isin(['AllErr','FN'])&s.contrast.isin(['G2-G1','P_full-P_base'])].sort_values(['cohort','target','contrast'])
    strictperf=sp[(sp.variant=='global')&(sp.learner=='additive')&sp.target.isin(['AllErr','FN'])]
    baselines=[];basedecomp=[]
    for base in ('freddie_mac_2018','fannie_mae_2018'):
        p=OUT/'historical_2018_baselines'/base
        d=pd.read_csv(p/'paired_contrasts.csv');baselines.append(d[d.target.isin(['AllErr','FN'])])
        basedecomp.append(pd.read_csv(p/'decomposition.csv'))
    baselines=pd.concat(baselines);basedecomp=pd.concat(basedecomp)
    choices=pd.read_csv(OUT/'development_policy_choices.csv')
    choices=choices[(choices.variant=='global')&(choices.learner=='additive')&choices.target.isin(['AllErr','FN'])&(choices.budget==.1)]

    # Scientific consistency checks precede all aggregate writes.
    assert len(main)==24
    assert len(strict)==12 and (strict.family_size==0).all() and strict.family_adjusted_lower.isna().all()
    for frame in (perf,sp):
        valid=frame.target_n>0
        assert np.allclose(frame.loc[valid,'capture'],frame.loc[valid,'hits']/frame.loc[valid,'target_n'])
        assert np.allclose(frame.yield_,frame.hits/frame.B)
        assert (frame.B==np.ceil(frame.budget*frame.N)).all()
        assert ((frame.hits>=0)&(frame.hits<=frame.target_n)&(frame.hits<=frame.B)).all()
    assert dec.identity_residual.abs().max()<1e-12
    receipt_count=0
    for p in OUT.rglob('receipt.json'):
        r=json.loads(p.read_text())
        for name,expected in r['results'].items():
            assert hashlib.sha256((p.parent/name).read_bytes()).hexdigest()==expected
        receipt_count+=1
    assert receipt_count==38

    for name,frame in [('main_historical_e2e3',main),('main_historical_performance',mainperf),
                       ('model_information_sensitivity',sensitivity),('main_predictive',mainpred),
                       ('main_budget_boundary',mainswap),('main_strict_sensitivity',strict),
                       ('main_strict_performance',strictperf)]:
        frame.to_csv(OUT/f'{name}.csv',index=False)
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    summary=dict(status='completed exploratory retrospective fixed-system analyses; independent evaluation NOT completed',
        created_at_utc=stamp,
        definitions={'primary':'global candidate floor, additive G models, b=.10, AllErr/FN, G2-G1/P_full-P_base',
            'family':'6 historical cohort cells x 2 targets x 2 contrasts = 24; Bonferroni approximate percentile intervals',
            'bootstrap':'9999 primary; 1999 other historical and2018;999 strict sensitivity',
            'endpoint':'historical recorded loan-age-based endpoint; strict sensitivity changes both outcome and eligible population',
            'consensus':'logit(mean candidate probabilities), clip1e-6',
            'log_loss_sign':'negative difference favors first model',
            'membership':'G2 entering/leaving relative to G1; paired counts are outcome-target-specific',
            'EM_information':'diagnostic target uses candidate-flip mask even for G0/P_score target conversion'},
        main_contrasts=records(main),main_performance=records(mainperf),main_log_loss=records(mainpred),
        main_membership=records(mainswap),information_model_sensitivity=records(sensitivity),
        strict_contrasts=records(strict),strict_performance=records(strictperf),
        development_choices=records(choices),historical_2018_contrasts=records(baselines),
        historical_2018_decomposition=records(basedecomp),
        interpretation_counts={'primary_comparisons':len(main),'positive_points':int((main.capture_difference>0).sum()),
            'pointwise_lower_above_zero':int((main.lower>0).sum()),
            'adjusted_lower_above_zero':int((main.family_adjusted_lower>0).sum()),
            'warning':'Counts are descriptive across related comparisons, not independent successes/trials.'},
        not_completed=['independent D_eval','candidate retraining/reconstruction variability',
                       'reviewer corrections or institutional benefit','prospective deployment'],
        execution_provenance={'historical_plan':'implementation_amendment_02/analysis_plan.json',
            'historical_code':'implementation_amendment_02/audit_revision_v1_9.py',
            'current_strict_and_guard_plan':'analysis_plan.json'})
    (OUT/'main_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    validation=dict(passed=True,tests=11,main_contrasts=24,strict_contrasts=12,
        execution_receipts_verified=receipt_count,historical_configurations=24,strict_configurations=12,
        baseline_cohorts=2,maximum_decomposition_residual=float(dec.identity_residual.abs().max()),
        independent_evaluation_completed=False,checked_at_utc=stamp)
    (OUT/'VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation,indent=2))


if __name__=='__main__': main()
