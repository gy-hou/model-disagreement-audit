import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import candidate_rebuild_v1_9 as rebuild
import build_reference_ensemble_v0_7 as training


def test_seed_override_replaces_config_seed_and_restores_loader():
    original=training.candidate_trainer.load_candidate_config
    for axis in ('M1','M2','M7'):
        template=training._axis_grid_specs(axis)[0]
        with training._single_candidate_grid(axis,template,rebuild.SEEDS[0]):
            item=training.candidate_trainer.load_candidate_config(axis)[0]
            assert item['random_state']==rebuild.SEEDS[0]
        assert training.candidate_trainer.load_candidate_config is original
    assert len(training._axis_grid_specs('M6'))==9


def test_global_gate_rejects_incomplete_or_single_member_grid():
    m=pd.DataFrame(dict(candidate_id=[f'x{i}' for i in range(26)],status='ok',validation_auc=[.8]+[.7]*25))
    keep,status=rebuild.eligible(m)
    assert keep.sum()==1 and status=='degenerate_M_below_2'
    m.loc[1,'validation_auc']=.79000001
    keep,status=rebuild.eligible(m)
    assert keep.sum()==2 and status=='ok'
    m.loc[2,'status']='failed'
    assert rebuild.eligible(m)[1]=='incomplete_grid'
    assert rebuild.eligible(m.iloc[:-1])[1]=='incomplete_grid'


def test_paired_comparison_undefined_empty_target_and_zero_identical_model():
    rng=np.random.default_rng(1);p=rng.uniform(.01,.3,200)
    z=dict(row_id=np.arange(200),y=np.zeros(200,int),p_dep=p,cand=np.c_[p*.8,p*1.2])
    rows,loss,_=rebuild.paired_results(z,z['cand'],dict(G1=p,G2=p),.15,32,boots=29)
    err,fn=rows
    assert err['capture_difference']==err['lower']==err['upper']==0
    assert np.isnan(fn['capture_difference']) and fn['bootstrap_empty_target']==29
    assert fn['status']=='undefined_empty_target'
    assert loss['difference']==loss['lower']==loss['upper']==0


def test_membership_exchange_explains_exact_hit_difference_and_budget():
    rng=np.random.default_rng(55);p=rng.uniform(.01,.3,101)
    z=dict(row_id=np.arange(101),y=rng.binomial(1,p),p_dep=p,cand=np.c_[p*.8,p*1.2])
    rows,_,members=rebuild.paired_results(z,z['cand'],dict(G1=p,G2=np.clip(p+rng.normal(0,.02,101),.001,.99)),.15,32,boots=19)
    for r in rows:
        assert r['B']==11
        assert r['entering_hits']-r['leaving_hits']==r['hits_difference']
        assert r['entering_n']==r['leaving_n']
        for q in members[r['target']].values():assert len(q)==len(np.unique(q))==11


def test_later_labels_do_not_change_ranked_membership():
    rng=np.random.default_rng(4);p=rng.uniform(.01,.2,120)
    z=dict(row_id=np.arange(120),y=rng.binomial(1,p),p_dep=p,cand=np.c_[p*.8,p*1.2])
    pred=dict(G1=p,G2=np.clip(p+.01,.001,.99))
    _,_,a=rebuild.paired_results(z,z['cand'],pred,.1,8,boots=3)
    _,_,b=rebuild.paired_results(dict(z,y=1-z['y']),z['cand'],pred,.1,8,boots=3)
    for target in rebuild.TARGETS:
        for g in ('G1','G2'):np.testing.assert_array_equal(a[target][g],b[target][g])


def test_worker_limit_applies_to_explicit_parallel_estimators():
    t=training.candidate_trainer
    with rebuild.limited_training_threads():
        assert t.RandomForestClassifier(n_jobs=-1).n_jobs==5
        assert t.ExplainableBoostingClassifier(n_jobs=-2).n_jobs==5
        x=t.build_xgb(training._axis_grid_specs('M2')[0],random_state=rebuild.SEEDS[0])
        assert x.n_jobs==5
