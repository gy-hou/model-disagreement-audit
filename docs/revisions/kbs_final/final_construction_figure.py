"""Verify archived construction results and redraw their final presentation.

This script reads completed experiments only. It draws no new labels, fits no
learner, builds no queue, and writes only new final-presentation artifacts.
The A/B/D organization follows src/mechanism_v1_10.py:figures; panel C now
reports the completed post-selection refit experiment on its expected scale.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/revisions/kbs_final"
FIG = ROOT / "paper/kbs_final/figures"
MECH = ROOT / "results/tables/v1_10_mechanism"
SOURCES = {}
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    SOURCES[str(path.relative_to(ROOT))] = digest(path)
    return path


def read(path):
    return pd.read_csv(source(path))


def mean_summary(g, column):
    values = g[column]
    mean = float(values.mean())
    half = 1.96 * values.std(ddof=1) / np.sqrt(len(g))
    return dict(mean=mean, mc_lo=float(mean-half), mc_hi=float(mean+half),
                q025=float(values.quantile(.025)), q975=float(values.quantile(.975)))


def verify_and_summarize():
    # Integrity of the existing v1.10 results, not a scientific rerun.
    receipt = json.loads(source(MECH / "run_manifest.json").read_text())
    for name, expected in receipt["outputs"].items():
        assert digest(source(MECH / name)) == expected, f"Archived result changed: {name}"
    source(ROOT / "src/mechanism_v1_10.py")
    source(ROOT / "idea-update/scripts/09_postgate_refit_sim.py")
    fr = read(ROOT / "idea-update/results/09_postgate_refit_sim.csv")
    current = read(MECH / "comparison_summary.csv")
    pop = read(MECH / "population_comparisons.csv")
    atoms = read(MECH / "population_atoms.csv")
    cm = read(MECH / "candidate_population_metrics.csv")
    hist = read(ROOT / "results/tables/v1_9_mechanism/mechanism_primary_summary.csv")
    old_cm = read(ROOT / "results/tables/v1_9_mechanism/candidate_population_metrics.csv")
    assert len(fr) == 1200 and not fr.duplicated(["set", "seed", "n_train"]).any()
    assert np.isfinite(fr.select_dtypes(include="number")).all().all()
    assert set(fr["set"]) == {"fixed3", "postgate"}
    assert set(fr.n_train) == {8000, 80000}
    assert np.allclose(fr.gain_pp, 100 * (fr.cap_G2-fr.cap_G1), atol=1e-12)
    expected_positives = float(np.dot(atoms.mass, atoms.q) * 10000)
    assert np.isclose(expected_positives, 622)
    assert cm.passes_global_auc_010.all()
    # Independent finite-support AUC identity: score ties contribute one half.
    pos = (atoms.mass * atoms.q).to_numpy()
    neg = (atoms.mass * (1-atoms.q)).to_numpy()
    for name in ("lower", "center", "upper"):
        s = atoms["candidate_"+name].to_numpy()
        wins = (s[:, None] > s[None, :]) + .5 * (s[:, None] == s[None, :])
        auc = (pos[:, None]*neg[None, :]*wins).sum()/(pos.sum()*neg.sum())
        recorded = cm.set_index("candidate").loc[name, "auc"]
        assert np.isclose(auc, recorded, atol=1e-12)
    assert cm.auc.max()-cm.auc.min() < .010
    at10 = pop[pop.budget == .10].set_index("target")
    assert np.isclose(at10.loc["Pos", "oracle_expected_extra_hits_10000"], 1.5)
    assert np.isclose(at10.loc["AllErr", "oracle_expected_extra_hits_10000"], 1.5)
    assert np.isclose(at10.loc["FN", "oracle_expected_extra_hits_10000"], 2.5)
    assert np.allclose(pop[pop.budget == .20].oracle_capture_gain, 0, atol=1e-12)
    exact = at10.loc["Pos"]
    summaries = []
    for (candidate_set, n), group in fr.groupby(["set", "n_train"]):
        assert len(group) == 300 and set(group.seed) == set(range(300))
        counts = group.M_ret.value_counts().to_dict()
        assert counts == {3: 164, 2: 111, 1: 25}
        assert set(group.kept) == {"LCU", "CU", "LC", "U", "L"}
        if candidate_set == "postgate":
            # The extra range representation is redundant for every retained
            # singleton or adjacent pair in this completed experiment.
            redundant = group[group.M_ret < 3]
            assert np.allclose(redundant[["gain_pp", "exp_gain_pp", "oracle_gain_pp"]], 0, atol=1e-12)
            assert np.allclose(group[group.M_ret == 3].oracle_gain_pp,
                               float(exact.oracle_capture_gain)*100, atol=1e-12)
        for column, estimand in [("gain_pp", "realized_label_capture"),
                                 ("exp_gain_pp", "expected_hits_over_expected_positives")]:
            stats = mean_summary(group, column)
            summaries.append(dict(candidate_set=candidate_set, n_train=int(n),
                                  estimand=estimand, repetitions=len(group), **stats,
                                  expected_extra_hits_per_10000=(stats["mean"]*expected_positives/100
                                      if column == "exp_gain_pp" else None),
                                  oracle_mean_pp=float(group.oracle_gain_pp.mean())))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "final_refit_summary.csv", index=False, float_format="%.17g")
    return hist, old_cm, current, pop, fr, summary


def render(hist, old_cm, current, pop, fr, summary):
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 9, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False})
    exact = pop[(pop.target == "Pos") & (pop.budget == .1)].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 7.2))
    fig.subplots_adjust(left=.14, right=.98, bottom=.27, top=.93, wspace=.57, hspace=.67)
    names = ["A_exact", "B_boundary_primary", "C_information_no_boundary", "D_control_approximation"]
    labels = ["A: no information", "B: boundary moves", "C: same boundary", "D: weak control", "E: qualified"]
    ax = axes[0, 0]
    ax.set_title("A  Candidate eligibility", loc="left", weight="bold", pad=15)
    ax.axis("off")
    for x, text, ha in [(0, "Probe", "left"), (.73, "M", "center"), (.98, "Info.¹", "right")]:
        ax.text(x, 1.01, text, weight="bold", ha=ha, transform=ax.transAxes)
    for i, label in enumerate(labels):
        y = .87-i*.16
        if i < 4:
            p = old_cm[(old_cm.scenario == names[i]) & old_cm.model.str.startswith("candidate_")]
            count = int(p.passes_population_global_auc_010.sum())
            info = float(hist[hist.scenario == names[i]].oracle_population_logloss_gain.iloc[0])
        else:
            count, info = 3, float(exact.conditional_information_nats)
        ax.text(0, y, label, transform=ax.transAxes, weight="bold" if i == 4 else "normal")
        ax.text(.73, y, f"{count}/3", color=GREEN if count == 3 else ORANGE,
                weight="bold", ha="center", transform=ax.transAxes)
        ax.text(.98, y, "yes" if info > 1e-12 else "no", ha="right", transform=ax.transAxes)
    ax.text(0, -.06, "Global AUC gap ≤ 0.010\nB/C/D: one passes; range is zero.\n¹Information in designed full sets.",
            transform=ax.transAxes, va="top", linespacing=1.4)

    ax = axes[0, 1]
    ax.set_title("B  E: oracle budget effect", loc="left", weight="bold", pad=15)
    pp = pop[pop.target == "Pos"].sort_values("budget")
    ax.plot(pp.budget*100, pp.oracle_capture_gain*100, "D-", color=GREEN, markersize=5, linewidth=1.5)
    ax.axhline(0, color=".6", linewidth=.8)
    ax.set(xticks=[5, 10, 15, 20], xlim=(4, 21), ylim=(-.02, .29),
           xlabel="Review budget (%)", ylabel="Oracle capture gain (pp)")
    ax.annotate("+1.5 expected hits\nper 10,000 cases", (10, float(exact.oracle_capture_gain)*100),
                xytext=(5.5, .12), arrowprops=dict(arrowstyle="-", color=".4", lw=.7))
    ax.annotate("Same queue", (20, 0), xytext=(14.5, .045),
                arrowprops=dict(arrowstyle="-", color=".4", lw=.7))

    ax = axes[1, 0]
    ax.set_title("C  E: gate, recompute, refit", loc="left", weight="bold", pad=15)
    fitted = summary[(summary.candidate_set == "postgate") &
                     (summary.estimand == "expected_hits_over_expected_positives")].set_index("n_train")
    for n, y in [(8000, 1), (80000, 0)]:
        r = fitted.loc[n]
        ax.hlines(y, r.q025, r.q975, color=".65", lw=2)
        ax.errorbar(r["mean"], y, xerr=[[r["mean"]-r.mc_lo], [r.mc_hi-r["mean"]]],
                    fmt="o", color=BLUE, markersize=6, elinewidth=4, capsize=4)
        ax.text(-.35, y+.31, f"Mean {r['mean']:+.3f} pp", fontsize=9,
                zorder=10, bbox=dict(facecolor="white", edgecolor="none", alpha=1, pad=1.5))
    ax.axvline(0, color=".6", lw=.8)
    ax.axvline(fitted.oracle_mean_pp.iloc[0], color=GREEN, linestyle="--", lw=1.2)
    ax.set(yticks=[0, 1], yticklabels=["80,000", "8,000"], ylim=(-.55, 1.8),
           xlim=(-.38, .62), xticks=[-.3, 0, .3, .6], ylabel="Training labels",
           xlabel="Expected capture gain (pp)")
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", lw=3, label="Mean ± 1.96 MCSE"),
        Line2D([], [], color=".65", lw=2, label="Central 95% repetitions"),
        Line2D([], [], color=GREEN, ls="--", label="Mean post-gate oracle")],
        loc="upper left", bbox_to_anchor=(-.12, -.27), frameon=False, handlelength=1.8)

    ax = axes[1, 1]
    ax.set_title("D  Fixed full-design probes", loc="left", weight="bold", pad=15)
    h = hist.set_index("scenario")
    for i, name in enumerate(names+["E_qualified"]):
        if i < 4:
            r = h.loc[name]
            oracle = r.oracle_population_capture_gain*100
            values = [(r.saturated_mean*100, r.saturated_mc_mean_lo*100, r.saturated_mc_mean_hi*100),
                      (r.additive_mean*100, r.additive_mc_mean_lo*100, r.additive_mc_mean_hi*100)]
        else:
            oracle = float(exact.oracle_capture_gain)*100
            values = []
            for learner in ("Saturated", "Additive"):
                r = current[(current.n_train == 8000) & (current.target == "Pos") &
                    (current.budget == .1) & (current.learner == learner) & (current.metric == "delta_capture")].iloc[0]
                values.append((r["mean"]*100, r.mc_mean_lo*100, r.mc_mean_hi*100))
        ax.plot(oracle, i-.19, "D", color="black", markersize=4)
        for j, (mu, lo, hi) in enumerate(values):
            ax.errorbar(mu, i+j*.19, xerr=[[mu-lo], [hi-mu]], fmt=["o", "s"][j],
                        color=[BLUE, ORANGE][j], markersize=4, elinewidth=1.1, capsize=2)
    ax.axvline(0, color=".65", lw=.8)
    ax.set(yticks=range(5), yticklabels=["A  3/3", "B  1/3*", "C  1/3*", "D  1/3*", "E  3/3"],
           xlabel="Capture gain (pp), fixed sets", xlim=(-.8, 4))
    ax.invert_yaxis()
    ax.legend(handles=[Line2D([], [], marker="D", color="black", ls="", label="Population oracle"),
        Line2D([], [], marker="o", color=BLUE, ls="", label="Saturated"),
        Line2D([], [], marker="s", color=ORANGE, ls="", label="Additive")],
        loc="upper left", bbox_to_anchor=(-.10, -.27), frameon=False, handlelength=1.3)
    fig.text(.05, .02, "Pos target; 10% budget in C/D; 300 repetitions per fitted experiment.\n"
             "C: post-gate refits, expected denominator 622 per 10,000 cases.\n"
             "D: fixed sets, 8,000 training labels. *Only one candidate is AUC-eligible.",
             fontsize=9, va="bottom", linespacing=1.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        if box.x0 < -.001 or box.y0 < -.001 or box.x1 > 1.001 or box.y1 > 1.001:
            clipped.append(text.get_text())
    assert not clipped, clipped
    FIG.mkdir(exist_ok=True)
    outputs = {}
    for ext in ("pdf", "svg", "png"):
        path = FIG / f"final_mechanisms.{ext}"
        fig.savefig(path, dpi=300, facecolor="white")
        outputs[str(path.relative_to(ROOT))] = digest(path)
    plt.close(fig)
    return outputs


def main():
    data = verify_and_summarize()
    outputs = render(*data)
    outputs[str((OUT/"final_refit_summary.csv").relative_to(ROOT))] = digest(OUT/"final_refit_summary.csv")
    assert all(digest(ROOT/path) == sha for path, sha in SOURCES.items())
    manifest = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                    script_sha256=digest(__file__), scientific_rerun=False,
                    source_sha256=SOURCES, outputs=outputs,
                    checks=dict(old_production_hashes_unchanged=True, source_hashes_unchanged=True,
                        exact_population_auc_recomputed=True, eligibility_and_budget_values_verified=True,
                        complete_refit_groups_verified=True, realization_identity_verified=True,
                        redundant_retained_sets_zero_gain=True, all_text_inside_canvas=True),
                    dimensions_inches=[6.3, 7.2], minimum_font_pt=9, png_dpi=300)
    (OUT/"final_construction_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"checks": manifest["checks"], "outputs": list(outputs)}, indent=2))


if __name__ == "__main__":
    main()
