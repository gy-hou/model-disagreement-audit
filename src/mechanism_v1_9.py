"""Finite-support mechanism examples: exact population facts and label simulations.

This module never reads loan-level data or replaces historical experiments.
Run --freeze before --run. The frozen protocol is local, not preregistration.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, xlogy
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/tables/v1_9_mechanism"
FIG = ROOT / "results/figures/v1_9_mechanism"
TARGETS = ("Pos", "FN", "FP", "AllErr", "EM")
BUDGETS = (.05, .10, .15, .20)
SEEDS = tuple(range(261019000, 261019300))
N_TRAIN, N_EVAL = 8000, 10000
HISTORICAL = tuple(ROOT / "results/tables/v1_4_score_control" / f for f in (
    "simulation_summary.csv", "simulation_summary_bins100.csv",
    "simulation_seeds.csv", "simulation_seeds_bins100.csv"))


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str
    mass: np.ndarray
    score: np.ndarray
    consensus: np.ndarray
    spread: np.ndarray
    q: np.ndarray
    tau: float
    u: np.ndarray
    note: str

    @property
    def candidates(self):
        return np.column_stack((self.consensus - self.spread / 2,
                                self.consensus,
                                self.consensus + self.spread / 2))


def scenarios():
    mass = np.array([.8, .05, .05, .1])
    score = np.array([.020, .039, .041, .200])
    spread = np.array([.002, .004, .004, .006])
    exact = Scenario("A_exact", "A", mass, score, score, spread, score,
                     .040, np.zeros(4), "Exact four-group counterexample; D is a function of S.")
    u = np.tile([-1., 1.], 4)
    sr, mr = np.repeat(score, 2), np.repeat(mass / 2, 2)
    noisy = Scenario("A_independent_jitter", "A", mr, sr, sr,
                     np.repeat(spread, 2) * (1 + .1 * u), sr, .040, u,
                     "Range jitter is independent of Y conditional on S; original reversal retained.")
    bm = np.repeat(np.array([.6, .2, .1, .1]) / 2, 2)
    bs = np.repeat([.04, .08, .11, .12], 2)
    # Symmetric candidates preserve arithmetic consensus exactly. D reveals U.
    dr = .004 + .002 * u
    b1 = Scenario("B_boundary_weak", "B", bm, bs, bs, dr, bs + .010 * u,
                  .100, u, "Extra outcome information changes the Pos 10% boundary; alpha=.010.")
    b2 = Scenario("B_boundary_primary", "B", bm, bs, bs, dr, bs + .025 * u,
                  .100, u, "Extra outcome information changes the Pos 10% boundary; alpha=.025.")
    cs = np.repeat([.02, .04, .06, .20], 2)
    cq = cs + .005 * u * (cs < .20)
    c = Scenario("C_information_no_boundary", "C", bm, cs, cs, dr, cq,
                 .100, u, "Extra information improves log-loss; at 10% all outcome oracle queues coincide.")
    ds = np.array([.04, .04, .12, .12])
    dc = np.array([.08, .16, .08, .16])
    interaction = np.array([1., -1., -1., 1.])
    d = Scenario("D_control_approximation", "D", np.full(4, .25), ds, dc,
                 .008 + .004 * interaction, expit(-3 + 2 * interaction),
                 .100, interaction,
                 "D is deterministic from (S,C); an additive control misses their interaction.")
    return (exact, noisy, b1, b2, c, d)


def target_probability(q, s: Scenario, target):
    pred = s.score >= s.tau
    fn, fp = (~pred) * q, pred * (1 - q)
    flip = (s.candidates.min(1) < s.tau) & (s.candidates.max(1) >= s.tau)
    return {"Pos": q, "FN": fn, "FP": fp, "AllErr": fn + fp,
            "EM": (fn + fp) * flip}[target]


def group_ids(*columns):
    # Rounding identifies intended exact support, not an approximate risk bin.
    return np.unique(np.round(np.column_stack(columns), 12), axis=0, return_inverse=True)[1]


def oracle_q(s, augmented=False):
    ids = group_ids(s.score, s.consensus, s.spread) if augmented else group_ids(s.score, s.consensus)
    total = np.bincount(ids, weights=s.mass)
    mean = np.bincount(ids, weights=s.mass * s.q) / total
    return mean[ids]


def top_fraction(score, mass, budget):
    """Population budget policy; equal scores receive uniform tie randomization."""
    score = np.round(np.asarray(score, float), 12)
    if not 0 <= budget <= 1 or not np.isclose(np.sum(mass), 1):
        raise ValueError("Invalid budget or probability masses")
    pi, left, cutoff = np.zeros(len(score)), float(budget), np.nan
    for value in np.unique(score)[::-1]:
        at = score == value
        take = min(float(mass[at].sum()), left)
        if take > 0:
            pi[at] = take / mass[at].sum()
            cutoff = float(value)
            left -= take
        if left <= 1e-12:
            break
    if not np.isclose(np.dot(pi, mass), budget, atol=1e-12):
        raise AssertionError("Budget mismatch")
    return pi, cutoff


def auc_population(prediction, q, mass):
    """Exact pairwise AUC, including half credit for tied score groups."""
    pos, neg = mass * q, mass * (1 - q)
    comp = (prediction[:, None] > prediction[None, :]).astype(float)
    comp += .5 * (prediction[:, None] == prediction[None, :])
    return float(np.sum(pos[:, None] * neg[None, :] * comp) / (pos.sum() * neg.sum()))


def brier_population(p, q, mass):
    return float(np.dot(mass, q * (1 - p) ** 2 + (1 - q) * p ** 2))


def logloss_population(p, q, mass):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.dot(mass, -xlogy(q, p) - xlogy(1 - q, 1 - p)))


def decompose_population(pi, eta, mass, bins):
    total = np.dot(mass, eta)
    if total == 0:
        return (np.nan,) * 3
    nk = np.bincount(bins, weights=mass)
    tk = np.bincount(bins, weights=mass * eta)
    qk = np.bincount(bins, weights=mass * pi)
    capture = float(np.dot(mass * pi, eta) / total)
    position = float(np.sum(qk * tk / nk) / total)
    return capture, position, capture - position


def basis(s, augmented=False):
    cols = [s.score, s.consensus] + ([s.spread] if augmented else [])
    scaled = []
    for col in cols:
        center = np.dot(s.mass, col)
        scale = np.sqrt(np.dot(s.mass, (col - center) ** 2))
        scaled.append((col - center) / scale if scale > 1e-12 else np.zeros_like(col))
    return np.column_stack((np.ones(len(s.mass)), *scaled))


def fit_additive(s, counts, successes, augmented=False):
    """Same penalized logistic learner/budget; G2 adds range as one covariate."""
    x = basis(s, augmented)
    n = counts.sum()
    ridge = 1e-6

    def fun(beta):
        eta = x @ beta
        loss = np.sum(counts * np.logaddexp(0, eta) - successes * eta) / n
        loss += ridge * np.dot(beta[1:], beta[1:]) / 2
        grad = x.T @ (counts * expit(eta) - successes) / n
        grad[1:] += ridge * beta[1:]
        return loss, grad

    init = np.zeros(x.shape[1])
    init[0] = np.log((successes.sum() + .5) / (n - successes.sum() + .5))
    fit = minimize(fun, init, jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9})
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f"Logistic fit failed for {s.name}: {fit.message}")
    return expit(x @ fit.x)


def fit_saturated(s, counts, successes, augmented=False):
    ids = group_ids(s.score, s.consensus, s.spread) if augmented else group_ids(s.score, s.consensus)
    n, z = np.bincount(ids, weights=counts), np.bincount(ids, weights=successes)
    return ((z + .5) / (n + 1))[ids]


def protocol_spec():
    return {
        "schema": "mechanism_v1_9.1", "seeds": list(SEEDS), "repetitions": len(SEEDS),
        "n_train": N_TRAIN, "n_eval": N_EVAL, "budgets": list(BUDGETS), "targets": list(TARGETS),
        "primary_mechanism_target": "Pos at budget .10; AllErr/FN/FP also reported; EM is diagnostic",
        "sampling": "Fixed support counts; independent Bernoulli training/evaluation labels. Random row order breaks ties independently of labels.",
        "seed_streams": "SeedSequence([seed, scenario_index, 1901]) with independent train, evaluation, permutation streams",
        "fit": "Saturated beta(.5,.5) group estimates; penalized additive logistic ridge=1e-6, maxiter=500, same inputs except added range",
        "position_bins": "Exact score support groups, fixed before labels; not the historical 50/100-bin diagnostic",
        "uncertainty": "300-replicate mean +/- 1.96 MCSE and central empirical 95% label-repetition interval; neither is a future-population CI",
        "monte_carlo_planning": {"R": 300, "se_for_rate_0_05": float(np.sqrt(.05 * .95 / 300)),
                                  "normal_95_halfwidth_at_rate_0_05": float(1.96 * np.sqrt(.05 * .95 / 300)),
                                  "purpose": "Mechanism mean/direction comparison; no precise 5% level or formal conditional-independence test claim"},
        "candidate_eligibility": "Designed probe ensemble is not filtered. Exact AUC/Brier and global .010 AUC flags disclosed. Only A_exact asserts identical AUC.",
        "sensitivity_population_only": {"tau": [.025, .040, .100], "near_threshold_mass": [.10, .20]},
        "scenario_list": [{"name": s.name, "kind": s.kind, "mass": s.mass.tolist(),
                           "score": s.score.tolist(), "consensus": s.consensus.tolist(),
                           "spread": s.spread.tolist(), "q": s.q.tolist(), "tau": s.tau,
                           "u": s.u.tolist(), "note": s.note} for s in scenarios()],
        "historical_inputs_sha256": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in HISTORICAL},
        "scope": "Synthetic mechanism extension only; no real-loan fitting, prospective evaluation, utility trial, or theorem novelty claim",
    }


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        raise FileExistsError("Protocol already exists; refusing silent replacement")
    spec = protocol_spec()
    payload = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "locally frozen before production label simulations; not public preregistration",
               "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
               "spec_sha256": sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
               "spec": spec}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"protocol": str(path), "spec_sha256": payload["spec_sha256"]}))


def verify_protocol():
    payload = json.loads((OUT / "protocol.json").read_text())
    current = sha256(json.dumps(protocol_spec(), sort_keys=True).encode()).hexdigest()
    if current != payload["spec_sha256"]:
        raise ValueError("Current simulation contract or historical inputs differ from frozen protocol")
    current_source = sha256(Path(__file__).read_bytes()).hexdigest()
    if current_source != payload["source_sha256"]:
        path = OUT / "source_amendments.json"
        amendments = json.loads(path.read_text()) if path.exists() else []
        if not any(a.get("source_sha256") == current_source and
                   a.get("protocol_spec_sha256") == payload["spec_sha256"] and
                   a.get("reason") for a in amendments):
            raise ValueError("Source differs from frozen version without an explicit recorded amendment")
    return payload


def exact_outputs():
    atoms, candidates, metrics, comparisons = [], [], [], []
    for s in scenarios():
        q0, q1 = oracle_q(s), oracle_q(s, True)
        bins = group_ids(s.score)
        pmat = np.column_stack((s.score, s.candidates))
        aucs = [auc_population(pmat[:, j], s.q, s.mass) for j in range(4)]
        briers = [brier_population(pmat[:, j], s.q, s.mass) for j in range(4)]
        for j, name in enumerate(("reference_deployment", "candidate_lower", "candidate_center", "candidate_upper")):
            candidates.append(dict(scenario=s.name, model=name, auc=aucs[j], brier=briers[j],
                                   auc_gap_from_best=max(aucs) - aucs[j],
                                   passes_population_global_auc_010=aucs[j] >= max(aucs) - .010 - 1e-12,
                                   brier_excess_vs_deployment=briers[j] - briers[0],
                                   eligibility_scope="population diagnostic only; no empirical selection imposed"))
        for j in range(len(s.mass)):
            atoms.append(dict(scenario=s.name, kind=s.kind, atom=j, mass=s.mass[j], score=s.score[j],
                              consensus=s.consensus[j], range=s.spread[j], u=s.u[j], q_true=s.q[j],
                              q_oracle_W=q0[j], q_oracle_WD=q1[j], tau=s.tau,
                              candidate_lower=s.candidates[j, 0], candidate_center=s.candidates[j, 1],
                              candidate_upper=s.candidates[j, 2]))
        # Population pseudo-true weak logistic limit isolates approximation error.
        qweak = fit_additive(s, s.mass, s.mass * s.q)
        qweak_d = fit_additive(s, s.mass, s.mass * s.q, True)
        for b in BUDGETS:
            for target in TARGETS:
                eta = target_probability(s.q, s, target)
                total = np.dot(s.mass, eta)
                rank = {"Oracle_W": target_probability(q0, s, target),
                        "Oracle_WD": target_probability(q1, s, target),
                        "Weak_W_limit": target_probability(qweak, s, target),
                        "Weak_WD_limit": target_probability(qweak_d, s, target),
                        "ThresholdProb": -np.round(abs(s.score - s.tau), 12), "Range": s.spread}
                policies = {name: top_fraction(value, s.mass, b)[0] for name, value in rank.items()}
                for name, pi in policies.items():
                    cap, pos, within = decompose_population(pi, eta, s.mass, bins)
                    hit = np.dot(s.mass * pi, eta)
                    metrics.append(dict(scenario=s.name, kind=s.kind, budget=b, target=target, policy=name,
                                        target_probability=total, expected_hits_per_10000=10000 * hit,
                                        capture_ratio_of_expectations=cap, yield_expected=hit / b,
                                        hits_per_1000_reviews=1000 * hit / b,
                                        position_capture=pos, within_capture=within,
                                        capture_status="defined" if total > 0 else "undefined_empty_target"))
                for a, base in (("Oracle_WD", "Oracle_W"), ("Weak_WD_limit", "Weak_W_limit"),
                                ("Range", "ThresholdProb")):
                    pa, pb = policies[a], policies[base]
                    ca, poa, wa = decompose_population(pa, eta, s.mass, bins)
                    cb, pob, wb = decompose_population(pb, eta, s.mass, bins)
                    hit_delta = np.dot(s.mass * (pa - pb), eta)
                    _, cut = top_fraction(rank[a], s.mass, b)
                    boundary_rhs = np.dot(s.mass * (pa - pb), eta - cut)
                    comparisons.append(dict(scenario=s.name, kind=s.kind, budget=b, target=target,
                                            comparison=f"{a}-minus-{base}", delta_capture=ca - cb,
                                            delta_expected_hits_per_10000=10000 * hit_delta,
                                            delta_position=poa - pob, delta_within=wa - wb,
                                            expected_exchange_mass=.5 * np.dot(s.mass, abs(pa - pb)),
                                            cutoff=cut, boundary_identity_rhs=boundary_rhs,
                                            logloss_gain_W_to_WD=logloss_population(q0, s.q, s.mass) - logloss_population(q1, s.q, s.mass),
                                            info_max_abs=q1.__sub__(q0).__abs__().max(),
                                            capture_status="defined" if total > 0 else "undefined_empty_target"))
    for name, rows in (("population_atoms", atoms), ("candidate_population_metrics", candidates),
                       ("population_metrics", metrics), ("population_comparisons", comparisons)):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    sensitivity = []
    for near in (.10, .20):
        for tau in (.025, .040, .100):
            mass = np.array([.9 - near, near / 2, near / 2, .1])
            original = scenarios()[0]
            s = Scenario(f"near{near}_tau{tau}", "A", mass, original.score,
                         original.consensus, original.spread, original.q, tau, original.u, "Population-only sensitivity")
            t = top_fraction(-np.round(abs(s.score - tau), 12), mass, .1)[0]
            r = top_fraction(s.spread, mass, .1)[0]
            for target in TARGETS:
                eta = target_probability(s.q, s, target)
                den = np.dot(mass, eta)
                sensitivity.append(dict(tau=tau, near_threshold_mass=near, budget=.1, target=target,
                                        target_probability=den, threshold_capture=np.dot(mass * t, eta) / den if den else np.nan,
                                        range_capture=np.dot(mass * r, eta) / den if den else np.nan,
                                        capture_status="defined" if den else "undefined_empty_target"))
    pd.DataFrame(sensitivity).to_csv(OUT / "population_threshold_mass_sensitivity.csv", index=False)


def exact_counts(s, n):
    counts = np.rint(s.mass * n).astype(int)
    if counts.sum() != n or not np.allclose(counts / n, s.mass):
        raise ValueError("Sample size does not preserve fixed atom masses")
    return counts


def empirical_decomposition(selected, target, bins):
    total = int(target.sum())
    if not total:
        return (np.nan,) * 3
    nk = np.bincount(bins)
    tk = np.bincount(bins, weights=target)
    qk = np.bincount(bins[selected], minlength=len(nk))
    cap = target[selected].sum() / total
    pos = np.sum(qk * tk / nk) / total
    return float(cap), float(pos), float(cap - pos)


def simulate_one(s, index, seed):
    streams = np.random.SeedSequence([seed, index, 1901]).spawn(3)
    train_rng, eval_rng, order_rng = [np.random.default_rng(x) for x in streams]
    counts = exact_counts(s, N_TRAIN)
    successes = train_rng.binomial(counts, s.q)
    # Training labels and evaluation labels have separate, recorded streams.
    qmodels = {"Oracle_W": oracle_q(s), "Oracle_WD": oracle_q(s, True),
               "Saturated_W": fit_saturated(s, counts, successes),
               "Saturated_WD": fit_saturated(s, counts, successes, True),
               "Additive_W": fit_additive(s, counts, successes),
               "Additive_WD": fit_additive(s, counts, successes, True)}
    atoms = order_rng.permutation(np.repeat(np.arange(len(s.mass)), exact_counts(s, N_EVAL)))
    y = eval_rng.binomial(1, s.q[atoms]).astype(bool)
    pred = s.score[atoms] >= s.tau
    flip = ((s.candidates.min(1) < s.tau) & (s.candidates.max(1) >= s.tau))[atoms]
    targets = {"Pos": y, "FN": ~pred & y, "FP": pred & ~y,
               "AllErr": pred != y, "EM": (pred != y) & flip}
    bins = group_ids(s.score)[atoms]
    logloss = {}
    for name, q in qmodels.items():
        p = np.clip(q[atoms], 1e-12, 1 - 1e-12)
        logloss[name] = float(np.mean(-xlogy(y, p) - xlogy(~y, 1 - p)))
    qmetrics, contrasts = [], []
    for target, truth in targets.items():
        ranks = {name: target_probability(q, s, target)[atoms] for name, q in qmodels.items()}
        ranks.update(ThresholdProb=-np.round(abs(s.score[atoms] - s.tau), 12), Range=s.spread[atoms])
        orders = {name: np.argsort(-np.round(value, 12), kind="stable") for name, value in ranks.items()}
        for b in BUDGETS:
            B = int(np.ceil(N_EVAL * b))
            queues = {name: order[:B] for name, order in orders.items()}
            captures = {}
            for name, queue in queues.items():
                cap, pos, within = empirical_decomposition(queue, truth, bins)
                hits = int(truth[queue].sum())
                captures[name] = (cap, pos, within, hits)
                qmetrics.append(dict(scenario=s.name, kind=s.kind, seed=seed, target=target, budget=b,
                                     policy=name, n=N_EVAL, B=B, target_n=int(truth.sum()), hits=hits,
                                     capture=cap, yield_rate=hits / B, position_capture=pos, within_capture=within,
                                     logloss=logloss.get(name, np.nan)))
            for a, base in (("Oracle_WD", "Oracle_W"), ("Saturated_WD", "Saturated_W"),
                            ("Additive_WD", "Additive_W"), ("Range", "ThresholdProb")):
                ca, pa, wa, ha = captures[a]
                cb, pb, wb, hb = captures[base]
                qa, qb = queues[a], queues[base]
                in_a, in_b = np.zeros(N_EVAL, bool), np.zeros(N_EVAL, bool)
                in_a[qa], in_b[qb] = True, True
                enter, leave = in_a & ~in_b, in_b & ~in_a
                eta = target_probability(s.q, s, target)[atoms]
                expected_hit_delta = float(np.sum(eta[enter]) - np.sum(eta[leave]))
                contrasts.append(dict(scenario=s.name, kind=s.kind, seed=seed, target=target, budget=b,
                                      comparison=f"{a}-minus-{base}", delta_capture=ca - cb,
                                      delta_hits=ha - hb, delta_yield=(ha - hb) / B,
                                      delta_position=pa - pb, delta_within=wa - wb,
                                      exchanged_cases=int(enter.sum()), boundary_target_hits_enter=int(truth[enter].sum()),
                                      boundary_target_hits_leave=int(truth[leave].sum()),
                                      conditional_expected_delta_hits=expected_hit_delta,
                                      logloss_gain=logloss.get(base, np.nan) - logloss.get(a, np.nan),
                                      target_n=int(truth.sum())))
    return qmetrics, contrasts


def summarize(frame, keys, columns):
    rows = []
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        prefix = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for col in columns:
            a = group[col].dropna().to_numpy(float)
            n = len(a)
            mean = a.mean() if n else np.nan
            se = a.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            rows.append({**prefix, "metric": col, "repetitions": len(group), "defined_repetitions": n,
                         "mean": mean, "mcse": se, "mc_mean_lo": mean - 1.96 * se,
                         "mc_mean_hi": mean + 1.96 * se,
                         "label_replication_q025": np.quantile(a, .025) if n else np.nan,
                         "label_replication_q975": np.quantile(a, .975) if n else np.nan,
                         "positive_fraction": float(np.mean(a > 1e-12)) if n else np.nan,
                         "negative_fraction": float(np.mean(a < -1e-12)) if n else np.nan})
    return pd.DataFrame(rows)


def historical_record():
    rows = []
    for bins, path in ((50, HISTORICAL[0]), (100, HISTORICAL[1])):
        f = pd.read_csv(path)
        f.insert(0, "historical_bins", bins)
        f.insert(1, "source", str(path.relative_to(ROOT)))
        f.insert(2, "status", "historical results retained unchanged; not rerun in v1.9")
        rows.append(f)
    pd.concat(rows, ignore_index=True).to_csv(OUT / "historical_simulation_summary_unchanged.csv", index=False)


def convergence():
    """Exact-design label convergence for the analytic example, separately seeded."""
    s = scenarios()[0]
    rows = []
    for n in (2000, 10000, 50000):
        counts = exact_counts(s, n)
        tpi = top_fraction(-np.round(abs(s.score - s.tau), 12), s.mass, .1)[0]
        rpi = top_fraction(s.spread, s.mass, .1)[0]
        for seed in SEEDS:
            rng = np.random.default_rng(np.random.SeedSequence([seed, n, 1902]))
            positive = rng.binomial(counts, s.q)
            errors = np.where(s.score >= s.tau, counts - positive, positive)
            for name, pi in (("ThresholdProb", tpi), ("Range", rpi)):
                rows.append(dict(n=n, seed=seed, policy=name, allerror_capture=np.dot(pi, errors) / errors.sum(),
                                 allerror_hits=int(np.dot(pi, errors)), target_n=int(errors.sum())))
    f = pd.DataFrame(rows)
    f.to_csv(OUT / "counterexample_convergence_repetitions.csv", index=False)
    summarize(f, ["n", "policy"], ["allerror_capture", "allerror_hits"]).to_csv(
        OUT / "counterexample_convergence_summary.csv", index=False)


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    pop = pd.read_csv(OUT / "population_comparisons.csv")
    sim = pd.read_csv(OUT / "simulation_comparison_summary.csv")
    names = ["A_independent_jitter", "B_boundary_primary", "C_information_no_boundary", "D_control_approximation"]
    labels = ["A\nNo info", "B\nCrossing", "C\nNo crossing", "D\nWeak control"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    oracle = pop[(pop.budget == .1) & (pop.target == "Pos") & (pop.comparison == "Oracle_WD-minus-Oracle_W")].set_index("scenario").loc[names]
    axes[0].bar(np.arange(4), oracle.logloss_gain_W_to_WD, color=["#64748b", "#2563eb", "#0d9488", "#c2410c"])
    axes[0].set_ylabel("Population log-loss improvement (nats)")
    axes[0].set_title("A  Information beyond (score, consensus)")
    width = .24
    for offset, (method, label, color) in enumerate([
            ("Oracle_WD-minus-Oracle_W", "Oracle risk", "#1e3a8a"),
            ("Saturated_WD-minus-Saturated_W", "Saturated fitted", "#0d9488"),
            ("Additive_WD-minus-Additive_W", "Additive fitted", "#d97706")]):
        f = sim[(sim.budget == .1) & (sim.target == "Pos") & (sim.comparison == method) & (sim.metric == "delta_capture")].set_index("scenario").loc[names]
        axes[1].bar(np.arange(4) + (offset - 1) * width, 100 * f["mean"], width=width,
                    yerr=196 * f.mcse, label=label, color=color, capsize=2)
    axes[1].set_ylabel("Positive-target capture gain (percentage points)")
    axes[1].set_title("B  Fixed 10% budget; mean ± 1.96 MCSE")
    axes[1].legend(fontsize=7, loc="upper right")
    c = pop[(pop.scenario == "C_information_no_boundary") & (pop.target == "Pos") & (pop.comparison == "Oracle_WD-minus-Oracle_W")]
    axes[2].plot(c.budget * 100, c.delta_capture * 100, "o-", color="#0d9488")
    axes[2].set_xlabel("Budget (%)")
    axes[2].set_ylabel("Population oracle capture gain (percentage points)")
    axes[2].set_title("C  Same information, different budget")
    for ax in axes[:2]:
        ax.set_xticks(range(4), labels, fontsize=8)
    for ax in axes:
        ax.axhline(0, color="#64748b", lw=.7)
        ax.spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"mechanism_information_boundary.{ext}", dpi=180)
    plt.close(fig)


def primary_summary():
    """One machine-readable bridge table, derived only from frozen result files."""
    pop = pd.read_csv(OUT / "population_comparisons.csv")
    sim = pd.read_csv(OUT / "simulation_comparison_summary.csv")
    pop = pop[(pop.target == "Pos") & (pop.budget == .1) &
              (pop.comparison == "Oracle_WD-minus-Oracle_W")].set_index("scenario")
    rows = []
    for s in scenarios():
        row = {"scenario": s.name, "kind": s.kind, "target": "Pos", "budget": .1,
               "oracle_population_capture_gain": pop.loc[s.name, "delta_capture"],
               "oracle_population_logloss_gain": pop.loc[s.name, "logloss_gain_W_to_WD"],
               "oracle_exchange_mass": pop.loc[s.name, "expected_exchange_mass"]}
        for label, comparison in (("saturated", "Saturated_WD-minus-Saturated_W"),
                                  ("additive", "Additive_WD-minus-Additive_W")):
            f = sim[(sim.scenario == s.name) & (sim.target == "Pos") & (sim.budget == .1) &
                    (sim.comparison == comparison) & (sim.metric == "delta_capture")].iloc[0]
            for metric in ("mean", "mc_mean_lo", "mc_mean_hi", "label_replication_q025", "label_replication_q975"):
                row[f"{label}_{metric}"] = f[metric]
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "mechanism_primary_summary.csv", index=False)


def refresh_artifacts():
    verify_protocol()
    primary_summary()
    figures()
    path = OUT / "run_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        manifest["artifact_refresh_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["artifact_refresh_note"] = "Presentation labels and derived bridge table only; simulated values unchanged."
        manifest["current_source_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
        manifest["files"] = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                             for p in sorted([*OUT.glob("*.csv"), *FIG.glob("*")])}
        path.write_text(json.dumps(manifest, indent=2) + "\n")


def run():
    if (OUT / "run_manifest.json").exists():
        raise FileExistsError("A completed production run exists; refusing to overwrite it. --figures refreshes derived presentation only.")
    protocol = verify_protocol()
    exact_outputs()
    historical_record()
    convergence()
    all_metrics, all_comparisons = [], []
    for index, s in enumerate(scenarios()):
        for seed in SEEDS:
            a, b = simulate_one(s, index, seed)
            all_metrics.extend(a)
            all_comparisons.extend(b)
        print(f"completed {s.name}: {len(SEEDS)} independent label repetitions", flush=True)
    metrics, contrasts = pd.DataFrame(all_metrics), pd.DataFrame(all_comparisons)
    metrics.to_csv(OUT / "simulation_queue_repetitions.csv", index=False)
    contrasts.to_csv(OUT / "simulation_comparison_repetitions.csv", index=False)
    summarize(metrics, ["scenario", "kind", "target", "budget", "policy"],
              ["capture", "hits", "yield_rate", "position_capture", "within_capture", "logloss"]).to_csv(
                  OUT / "simulation_queue_summary.csv", index=False)
    summarize(contrasts, ["scenario", "kind", "target", "budget", "comparison"],
              ["delta_capture", "delta_hits", "delta_yield", "delta_position", "delta_within",
               "exchanged_cases", "conditional_expected_delta_hits", "logloss_gain"]).to_csv(
                   OUT / "simulation_comparison_summary.csv", index=False)
    primary_summary()
    figures()
    # Every historical source must remain unchanged, including the seed-level results.
    verify_protocol()
    manifest = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_spec_sha256": protocol["spec_sha256"],
                "current_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scenario_repetitions": len(scenarios()) * len(SEEDS),
                "queue_rows": len(metrics), "comparison_rows": len(contrasts),
                "historical_unchanged": True,
                "files": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                          for p in sorted([*OUT.glob("*.csv"), *FIG.glob("*")])}}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "completed", "repetitions": manifest["scenario_repetitions"],
                      "queue_rows": len(metrics), "comparison_rows": len(contrasts)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else refresh_artifacts() if args.figures else run()
