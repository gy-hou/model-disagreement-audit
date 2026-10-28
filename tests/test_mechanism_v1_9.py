"""Tests of scientific identities and design boundaries, not empirical success."""
import itertools
import json
from hashlib import sha256
import numpy as np
import pytest
from src import mechanism_v1_9 as m


def test_exact_counterexample_population_arithmetic():
    s = m.scenarios()[0]
    assert np.dot(s.mass, s.q) == pytest.approx(.04)
    et = m.target_probability(s.q, s, "AllErr")
    assert np.dot(s.mass, et) == pytest.approx(.1459)
    qt, _ = m.top_fraction(-abs(s.score-s.tau), s.mass, .1)
    qr, _ = m.top_fraction(s.spread, s.mass, .1)
    np.testing.assert_array_equal(qt, [0, 1, 1, 0])
    np.testing.assert_array_equal(qr, [0, 0, 0, 1])
    assert np.dot(s.mass*qt, et)/np.dot(s.mass, et) == pytest.approx(.0499/.1459)
    assert np.dot(s.mass*qr, et)/np.dot(s.mass, et) == pytest.approx(.08/.1459)
    auc = [m.auc_population(p, s.q, s.mass) for p in s.candidates.T]
    np.testing.assert_allclose(auc, .7292317708333333, atol=1e-14)
    base = m.brier_population(s.score, s.q, s.mass)
    assert base == pytest.approx(.0355199)
    assert m.brier_population(s.candidates[:, 0], s.q, s.mass)-base == pytest.approx(.0000021)
    assert m.brier_population(s.candidates[:, 2], s.q, s.mass)-base == pytest.approx(.0000021)


def test_valid_candidates_consensus_and_probability_contract():
    for s in m.scenarios():
        assert np.all((s.candidates > 0) & (s.candidates < 1))
        assert np.all((s.q > 0) & (s.q < 1))
        np.testing.assert_allclose(s.candidates.mean(1), s.consensus, atol=1e-15)
        np.testing.assert_allclose(np.ptp(s.candidates, axis=1), s.spread, atol=1e-15)


def test_information_and_budget_boundary_are_distinct():
    for s in m.scenarios():
        q0, q1 = m.oracle_q(s), m.oracle_q(s, True)
        pi0, _ = m.top_fraction(q0, s.mass, .1)
        pi1, c = m.top_fraction(q1, s.mass, .1)
        gain = np.dot(s.mass*(pi1-pi0), s.q)
        assert gain >= -1e-12
        assert gain == pytest.approx(np.dot(s.mass*(pi1-pi0), q1-c))
        llgain = m.logloss_population(q0,s.q,s.mass)-m.logloss_population(q1,s.q,s.mass)
        if s.kind in ("A", "D"):
            np.testing.assert_allclose(q0,q1)
            assert gain == pytest.approx(0)
        elif s.kind == "B":
            assert gain > 0 and llgain > 0
        else:
            assert llgain > 0
            np.testing.assert_array_equal(pi0,pi1)
            assert gain == pytest.approx(0)
            for target in ("Pos", "FN", "FP", "AllErr"):
                p0,_=m.top_fraction(m.target_probability(q0,s,target),s.mass,.1)
                p1,_=m.top_fraction(m.target_probability(q1,s,target),s.mass,.1)
                np.testing.assert_array_equal(p0,p1)


def test_d_control_approximation_has_no_unique_information():
    s=m.scenarios()[-1]
    weak=m.fit_additive(s,s.mass,s.mass*s.q)
    aug=m.fit_additive(s,s.mass,s.mass*s.q,True)
    assert m.logloss_population(weak,s.q,s.mass)>m.logloss_population(aug,s.q,s.mass)+.01
    np.testing.assert_allclose(m.oracle_q(s),m.oracle_q(s,True))
    count=m.exact_counts(s,8000)
    success=np.array([501,10,15,510])
    np.testing.assert_array_equal(m.fit_saturated(s,count,success),m.fit_saturated(s,count,success,True))


def test_top_b_and_stability_bound_by_brute_force():
    eta=np.array([.1,.2,.4,.7,.9])
    estimates=[eta+np.array(e)*.08 for e in itertools.product([-1,1],repeat=5)]
    for B in (1,2,3):
        opt=max(sum(eta[list(q)]) for q in itertools.combinations(range(5),B))
        for hat in estimates:
            selected=np.argsort(-hat,kind="stable")[:B]
            assert opt-eta[selected].sum() <= 2*B*np.max(abs(hat-eta))+1e-12


def test_queue_decomposition_common_denominator_and_empty_target():
    z=np.array([1,0,0,1,1,0],bool)
    bins=np.array([0,0,0,1,1,1])
    a=m.empirical_decomposition(np.array([0,1]),z,bins)
    b=m.empirical_decomposition(np.array([3,4]),z,bins)
    assert a[0]-b[0] == pytest.approx((a[1]-b[1])+(a[2]-b[2]))
    assert all(np.isnan(x) for x in m.empirical_decomposition(np.array([0,1]),np.zeros(6,bool),bins))


def test_population_ties_do_not_favor_unobserved_information():
    pi,c=m.top_fraction(np.array([.1,.2,.2]),np.array([.6,.2,.2]),.1)
    np.testing.assert_array_equal(pi,[0,.25,.25])
    assert c == .2


def test_completed_production_refuses_overwrite(tmp_path, monkeypatch):
    (tmp_path / "run_manifest.json").write_text("{}")
    monkeypatch.setattr(m, "OUT", tmp_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        m.run()


def test_unrecorded_source_change_fails_closed(tmp_path, monkeypatch):
    source = tmp_path / "changed_source.py"
    source.write_text("# Unrecorded change to statistical code")
    spec = m.protocol_spec()
    (tmp_path / "protocol.json").write_text(json.dumps({
        "spec_sha256": sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "source_sha256": "original-source-different-from-current"}))
    monkeypatch.setattr(m, "OUT", tmp_path)
    monkeypatch.setattr(m, "__file__", str(source))
    with pytest.raises(ValueError, match="without an explicit recorded amendment"):
        m.verify_protocol()
