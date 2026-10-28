"""Tables, figures and prose numbers for manuscript v1.11, generated only from completed result files.

No estimation is performed: inputs are the idea-update result CSVs, the v1.9/v1.10 aggregate CSVs and
exact rational evaluation of the construction in idea-update/scripts/00_exact_construction.py.
Outputs: paper/v1_11/generated/*.tex (tables, numbers.tex) and paper/v1_11/figures/*.{pdf,png}.
"""
import contextlib, importlib.util, io, json, math
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IU = ROOT / 'idea-update/results'
OUT = ROOT / 'paper/v1_11'
GEN, FIG = OUT / 'generated', OUT / 'figures'
REV = ROOT / 'results/tables/v1_9_revision'
MECH = ROOT / 'results/tables/v1_10_mechanism'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#9a9891'
VALS = {}


def val(key, v):
    VALS[key] = v
    return v


def s(x, nd=0, plus=True):
    """Signed number for LaTeX text/tables (math minus)."""
    if isinstance(x, float) and nd == 0:
        x = int(round(x))
    txt = f'{abs(x):,.{nd}f}' if nd else f'{abs(int(x)):,}'
    if x > 0:
        return ('$+$' if plus else '') + txt
    if x < 0:
        return '$-$' + txt
    return txt


def f(x, nd=1, pct=False):
    """Unsigned-style number with a math minus for negatives; avoids negative zero."""
    v = round(float(x), nd)
    if v == 0: v = 0.0
    txt = f'{abs(v):,.{nd}f}' + ('\\%' if pct else '')
    return ('$-$' + txt) if v < 0 else txt


def ci(lo, hi, nd=0):
    return f'[{s(lo, nd)}, {s(hi, nd)}]'


def tab(cols, rows, align, caption=None, label=None, size='\\small', longtable=False, notes=None, sep='3pt'):
    head = ' & '.join(cols) + r'\\'
    body = '\n'.join(' & '.join(str(c) for c in r) + r'\\' for r in rows)
    if longtable:
        t = (f'{{{size}\\setlength{{\\tabcolsep}}{{{sep}}}\\begin{{longtable}}{{{align}}}\n'
             + (f'\\caption{{{caption}}}\\label{{{label}}}\\\\\n' if caption else '')
             + f'\\toprule\n{head}\n\\midrule\\endfirsthead\n\\toprule\n{head}\n\\midrule\\endhead\n{body}\n\\bottomrule\n\\end{{longtable}}}}')
        return t + (f'\n{{\\footnotesize {notes}}}' if notes else '')
    return f'\\begin{{tabular}}{{{align}}}\\toprule\n{head}\n\\midrule\n{body}\n\\bottomrule\\end{{tabular}}'


def write(name, text):
    (GEN / f'{name}.tex').write_text(text + '\n')


def load_exact():
    spec = importlib.util.spec_from_file_location('ex', ROOT / 'idea-update/scripts/00_exact_construction.py')
    ex = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(ex)
    return ex


# ---------------------------------------------------------------- theory
def theory(ex):
    cells, _ = ex.build(Fr(1, 2), Fr(1, 10), Fr(1, 10))
    sym = {}
    for c in cells:
        sym.setdefault((c['S'], tuple(sorted((c['p1'], c['p2'])))), []).append(c)
    rows = []
    for c in sorted(cells, key=lambda c: (c['p1'], c['p2'])):
        g = sym[(c['S'], tuple(sorted((c['p1'], c['p2']))))]
        e_sym = sum(x['err'] for x in g) / len(g)
        rows.append([f"{float(c['p1']):.1f}", f"{float(c['p2']):.1f}", f"{float(c['eta']):.1f}",
                     'positive' if c['yhat'] else 'negative', f"{float(c['S']):.4f}", f"{float(c['err']):.1f}", f"{float(e_sym):.2f}"])
    write('tab_construction', tab(['$P_1$', '$P_2$', r'$\eta_\varepsilon$', r'$\widehat y$', '$S$', 'Error prob.\\ (ordered)', 'Error prob.\\ (symmetric)'],
                                  rows, 'rrrlrrr'))
    b = Fr(1, 10)
    info = {'S,C': lambda c: (c['S'], c['p1'] + c['p2']),
            'S,C,D': lambda c: (c['S'], c['p1'] + c['p2'], abs(c['p1'] - c['p2'])),
            'sym': lambda c: (c['S'], tuple(sorted((c['p1'], c['p2'])))),
            'id': lambda c: (c['S'], c['p1'], c['p2'])}
    for tag, (pi, d, eps) in {'a': (Fr(1, 2), Fr(1, 10), Fr(1, 10)), 'b': (Fr(1, 25), Fr(1, 100), Fr(1, 100))}.items():
        cl, _ = ex.build(pi, d, eps)
        for k, f in info.items():
            val(f'th-{tag}-V-{k}', f'{float(10000 * ex.oracle_value(cl, f, b)):,.0f}')
        emass = sum(c['m'] * c['err'] for c in cl)
        dv = ex.oracle_value(cl, info['id'], b) - ex.oracle_value(cl, info['sym'], b)
        val(f'th-{tag}-errmass', f'{float(emass):.4f}'); val(f'th-{tag}-dcap', f'{100 * float(dv / emass):.2f}')
        val(f'th-{tag}-dprec', f'{100 * float(dv / b):.1f}'); val(f'th-{tag}-dV10k', f'{float(10000 * dv):,.0f}')
        auc = ex.auc(cl, 'p1')
        val(f'th-{tag}-auc', f'{auc.numerator}/{auc.denominator}')
        val(f'th-{tag}-aucdec', f'{float(auc):.6f}')
        for k in ('p1', 'S'):
            val(f'th-{tag}-brier-{k}', f'{float(ex.brier(cl, k)):.6f}')
    # four-group counterexample (queue differences without information), exact
    mass = [Fr(80, 100), Fr(5, 100), Fr(5, 100), Fr(10, 100)]; S4 = [Fr(20, 1000), Fr(39, 1000), Fr(41, 1000), Fr(200, 1000)]; tau = Fr(40, 1000)
    err = [q if q < tau else 1 - q for q in S4]; emass = sum(m * e for m, e in zip(mass, err))
    thr = mass[1] * err[1] + mass[2] * err[2]; rng_ = mass[3] * err[3]
    val('cx-errmass', f'{float(emass):.4f}'); val('cx-cap-thr', f'{100 * float(thr / emass):.4f}\\%'); val('cx-cap-range', f'{100 * float(rng_ / emass):.4f}\\%')
    # figure data: Delta V vs epsilon at b = 1/10, and vs b at eps = 1/10 (pi = 1/2, d = 1/10)
    eps_grid = [Fr(k, 100) for k in range(-40, 41, 2)]
    dv_eps = []
    for e in eps_grid:
        if abs(e) > Fr(4, 10):
            continue
        cl, _ = ex.build(Fr(1, 2), Fr(1, 10), e)
        dv_eps.append((float(e), float(10000 * (ex.oracle_value(cl, info['id'], b) - ex.oracle_value(cl, info['sym'], b)))))
    b_grid = [Fr(k, 200) for k in range(0, 61)]
    cl, _ = ex.build(Fr(1, 2), Fr(1, 10), Fr(1, 10))
    dv_b = [(float(bb), float(10000 * (ex.oracle_value(cl, info['id'], bb) - ex.oracle_value(cl, info['sym'], bb))) if bb > 0 else 0.)
            for bb in b_grid]
    return dv_eps, dv_b


def fig_theory(dv_eps, dv_b):
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.5))
    e, v = zip(*dv_eps)
    ax[0].plot(e, v, color=BLUE, lw=2, solid_capstyle='round')
    ax[0].set_xlabel(r'perturbation $\varepsilon$'); ax[0].set_ylabel('extra expected hits per 10,000\n(budget b = 0.10)')
    ax[0].text(0, 408, r'$\varepsilon<0$: best ordered cell $(\pi+d,\,\pi-d)$', fontsize=6.8, color=INK2, ha='center', va='center')
    ax[0].text(0, 372, r'$\varepsilon>0$: best ordered cell $(\pi-d,\,\pi+d)$', fontsize=6.8, color=INK2, ha='center', va='center')
    ax[0].set_ylim(0, 432)
    bb, vb = zip(*dv_b)
    ax[1].plot(bb, vb, color=BLUE, lw=2, solid_capstyle='round')
    ax[1].axvline(1 / 9, color=GRAY, lw=1)
    ax[1].text(1 / 9 + .004, 20, r'$b = 1/9$', fontsize=7.5, color=INK2)
    ax[1].set_xlabel('budget fraction $b$'); ax[1].set_ylabel('extra expected hits per 10,000\n' + r'($\varepsilon = 0.1$)')
    for a, t in zip(ax, ('A', 'B')):
        a.set_title(t, loc='left', fontsize=9, fontweight='bold', color=INK)
        a.grid(color=GRID, lw=.6); a.set_axisbelow(True)
        for sp in ('top', 'right'): a.spines[sp].set_visible(False)
        a.tick_params(labelsize=7.5, colors=INK2); a.xaxis.label.set_size(8); a.yaxis.label.set_size(8)
    fig.tight_layout(); save(fig, 'fig_theory')


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=300, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------- synthetic evidence
SETTING_LABEL = {'pi.5_eps.1': r'$\pi{=}.5,\ \varepsilon{=}.1$', 'pi.5_eps.05': r'$\pi{=}.5,\ \varepsilon{=}.05$',
                 'pi.04_eps.01': r'$\pi{=}.04,\ \varepsilon{=}.01$'}


def positive_control():
    pc = pd.read_csv(IU / '11_identity_positive_control.csv')
    rows, summ = [], []
    for (st, n, lr), g in pc.groupby(['setting', 'n_dev', 'learner'], sort=False):
        m, sd, k = g.exp_gain_hits.mean(), g.exp_gain_hits.std(ddof=1), len(g)
        orc = g.oracle_hits_per10k.iloc[0]
        lo, hi = m - 1.96 * sd / math.sqrt(k), m + 1.96 * sd / math.sqrt(k)
        summ.append(dict(setting=st, n=n, learner=lr, oracle=orc, mgain=m, lo=lo, hi=hi, share=m / orc, pos=(g.exp_gain_hits > 0).mean(), k=k))
    S = pd.DataFrame(summ)
    for _, r in S.iterrows():
        key = f"pc-{r.setting}-{'hgb' if r.learner == 'interaction' else 'add'}-{int(r.n / 1000)}k"
        val(key + '-share', f(100 * r.share, 0, True)); val(key + '-mean', f(r.mgain))
        val(key + '-ci', f'[{f(r.lo)}, {f(r.hi)}]'); val(key + '-pos', f(100 * r.pos, 0, True))
        rows.append([SETTING_LABEL[r.setting], f'{r.oracle:.0f}', f'{int(r.n):,}', 'HGB' if r.learner == 'interaction' else 'additive',
                     f(r.mgain), f'[{f(r.lo)}, {f(r.hi)}]', f(100 * r.share, 0, True), f(100 * r.pos, 0, True)])
    write('s_poscontrol', tab(['Construction', 'Oracle gain', 'Dev.\\ $n$', 'Learner', 'Mean gain', 'MC 95\\% interval', 'Recovered', 'Reps $>0$'], rows, 'lrrlrrrr'))
    hgb = S[S.learner == 'interaction']; add = S[S.learner == 'additive']
    val('pc-hgb-share-min', f(100 * hgb.share.min(), 0, True)); val('pc-hgb-share-max', f(100 * hgb.share.max(), 0, True))
    val('pc-add-share-min', f(100 * add.share.min(), 0, True)); val('pc-add-share-max', f(100 * add.share.max(), 0, True))
    val('pc-reps', f'{int(S.k.iloc[0])}')
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.2), sharey=True)
    order = ['pi.5_eps.1', 'pi.5_eps.05', 'pi.04_eps.01']
    for ax, n in zip(axes, (10000, 80000)):
        for j, (lr, col, lab) in enumerate((('interaction', BLUE, 'HGB (interaction)'), ('additive', ORANGE, 'additive-logit'))):
            d = S[(S.n == n) & (S.learner == lr)].set_index('setting').loc[order]
            y = np.arange(len(order)) + (-.12 if j == 0 else .12)
            ax.errorbar(d['share'], y, xerr=[d['share'] - d['lo'] / d['oracle'], d['hi'] / d['oracle'] - d['share']], fmt='o', ms=5,
                        color=col, mec='white', mew=1, elinewidth=1.4, capsize=0, label=lab)
        ax.axvline(0, color=GRAY, lw=1); ax.axvline(1, color=GRAY, lw=1, ls=(0, (1, 2)))
        ax.set_title(f'development n = {n:,}', fontsize=8, color=INK, loc='left')
        ax.set_xlabel('share of oracle gain recovered', fontsize=8); ax.set_xlim(-.6, 1.25)
        ax.grid(axis='x', color=GRID, lw=.6); ax.set_axisbelow(True); ax.tick_params(labelsize=7.5, colors=INK2)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    oracle = S.drop_duplicates('setting').set_index('setting').loc[order, 'oracle']
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f'{SETTING_LABEL[o]}\noracle {oracle[o]:.0f}/10k' for o in order], fontsize=7.5)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, .9)); save(fig, 'fig_poscontrol')
    return S


def construction_e():
    fr = pd.read_csv(IU / '09_postgate_refit_sim.csv')
    rows = []
    for (st, n), g in fr.groupby(['set', 'n_train']):
        k = len(g); m, sd = g.exp_gain_pp.mean(), g.exp_gain_pp.std(ddof=1); mr, sr = g.gain_pp.mean(), g.gain_pp.std(ddof=1)
        lo, hi = m - 1.96 * sd / math.sqrt(k), m + 1.96 * sd / math.sqrt(k)
        key = f"F-{st}-{int(n / 1000)}k"
        val(key + '-exp', f'{m:+.3f}'.replace('+', '$+$').replace('-', '$-$')); val(key + '-expci', f'[{lo:.3f}, {hi:.3f}]'.replace('-', '$-$'))
        val(key + '-per10k', f(m * 622 / 100, 2)); val(key + '-oracle', f'{g.oracle_gain_pp.mean():.3f}')
        val(key + '-oracle10k', f'{g.oracle_gain_pp.mean() * 622 / 100:.2f}')
        val(key + '-real', f'{mr:+.3f}'.replace('+', '$+$').replace('-', '$-$'))
        val(key + '-realci', f'[{mr - 1.96 * sr / math.sqrt(k):.3f}, {mr + 1.96 * sr / math.sqrt(k):.3f}]'.replace('-', '$-$'))
        rows.append(['fixed three-candidate set' if st == 'fixed3' else 'after validation gate, refitted', f'{int(n):,}', k,
                     f'{g.oracle_gain_pp.mean():.3f}', f'{m:.3f}', f'[{lo:.3f}, {hi:.3f}]', f'{m * 622 / 100:.2f}',
                     f'{mr:.3f}', f'[{mr - 1.96 * sr / math.sqrt(k):.3f}, {mr + 1.96 * sr / math.sqrt(k):.3f}]'])
    rows = [[c.replace('-', '$-$') if isinstance(c, str) and c[:1] in '-[' else c for c in r] for r in rows]
    write('s_constructionE', tab(['Candidate set', 'Train labels', 'Reps', 'Oracle (pp)', 'Expected gain (pp)', 'MC 95\\%', 'Per 10,000',
                                  'Realized gain (pp)', 'MC 95\\%'], rows, 'lrrrrrrrr'))
    pg = fr[(fr.set == 'postgate') & (fr.n_train == 8000)]
    vc = pg.M_ret.value_counts().to_dict()
    for m_ in (1, 2, 3): val(f'F-mret-{m_}', str(vc.get(m_, 0)))
    val('F-reps', str(len(pg)))
    v = pd.read_csv(MECH / 'validation_summary.csv').set_index('event')
    for m_ in (1, 2, 3): val(f'E-v110-mret-{m_}', str(int(v.loc[f'M_{m_}', 'count'])))
    cpm = pd.read_csv(MECH / 'candidate_population_metrics.csv')
    val('E-auc-lo', f'{cpm.auc.min():.6f}'); val('E-auc-hi', f'{cpm.auc.max():.6f}'); val('E-auc-gap', f'{cpm.auc_gap_from_best.max():.4f}')
    pcmp = pd.read_csv(MECH / 'population_comparisons.csv').set_index(['target', 'budget'])
    val('E-pos10-hits', f"{pcmp.loc[('Pos', .10), 'oracle_expected_extra_hits_10000']:.1f}")
    val('E-pos10-pp', f"{100 * pcmp.loc[('Pos', .10), 'oracle_capture_gain']:.3f}")
    val('E-pos20-hits', f"{pcmp.loc[('Pos', .20), 'oracle_expected_extra_hits_10000']:.1f}")
    val('E-ET', '622')


# ---------------------------------------------------------------- mortgage
COH = [('freddie_mac', y) for y in (2021, 2022, 2023)] + [('fannie_mae', y) for y in (2021, 2022, 2023)]
CNAME = {'freddie_mac': 'Freddie', 'fannie_mae': 'Fannie'}


def mortgage():
    e2 = pd.read_csv(REV / 'main_historical_e2e3.csv')
    e2 = e2[(e2.scope == 'exploratory_retrospective') & (e2.variant == 'global') & (e2.learner == 'additive') & (e2.budget == .1)]
    perf = pd.read_csv(REV / 'main_historical_performance.csv')
    perf = perf[(perf.scope == 'exploratory_retrospective') & (perf.variant == 'global') & (perf.learner == 'additive') & (perf.budget == .1)]
    st2 = pd.read_csv(IU / '02_contrasts.csv')
    ff = pd.read_csv(IU / '05_feasible_headroom.csv')
    rows, g2, idr_a, idr_h, gx = [], [], [], [], []
    for inst, yr in COH:
        coh = f'{inst}_{yr}'
        for t in ('AllErr', 'FN'):
            a = e2[(e2.cohort == coh) & (e2.target == t) & (e2.contrast == 'G2-G1')].iloc[0]
            g1 = perf[(perf.cohort == coh) & (perf.target == t) & (perf.strategy == 'G1')].iloc[0]
            T = int(a.target_n)
            r1 = st2[(st2.inst == inst) & (st2.year == yr) & (st2.target == t) & (st2.variant == 'global') & (st2.learner == 'additive') & (st2.contrast == 'ID-R')].iloc[0]
            r2 = st2[(st2.inst == inst) & (st2.year == yr) & (st2.target == t) & (st2.variant == 'global') & (st2.learner == 'interaction') & (st2.contrast == 'ID-R')].iloc[0]
            r3 = ff[(ff.inst == inst) & (ff.year == yr) & (ff.target == t) & (ff.new == 'GX_large') & (ff.ref == 'G1_add')].iloc[0]
            g2.append(a.hits_difference); idr_a.append(r1.d_hits); idr_h.append(r2.d_hits); gx.append(r3.d_hits)
            rows.append([f'{CNAME[inst]} {yr}', t, f'{T:,}', f'{int(g1.hits):,}',
                         f'{s(a.hits_difference)} {ci(a.family_adjusted_lower * T, a.family_adjusted_upper * T)}',
                         f'{s(r1.d_hits)} {ci(r1.lo_pp / 100 * r1["T"], r1.hi_pp / 100 * r1["T"])}',
                         f'{s(r2.d_hits)} {ci(r2.lo_pp / 100 * r2["T"], r2.hi_pp / 100 * r2["T"])}',
                         f'{s(r3.d_hits)} {ci(r3.lo_hits, r3.hi_hits)}'])
    write('tab_mortgage', tab(['Cohort', 'Target', '$T$', 'G1 hits', 'G2$-$G1 [adj.\\ 24]', 'ID$-$R additive [adj.\\ 12]', 'ID$-$R HGB [95\\%]',
                               'G$_X-$G1 [95\\%]'], rows, 'llrrllll'))
    val('mo-g2-min', s(min(g2))); val('mo-g2-max', s(max(g2)))
    val('mo-g2-poslow', str(int((e2[e2.contrast == 'G2-G1'].family_adjusted_lower > 0).sum())))
    val('mo-idra-min', s(min(idr_a))); val('mo-idra-max', s(max(idr_a)))
    val('mo-idrh-min', s(min(idr_h))); val('mo-idrh-max', s(max(idr_h)))
    val('mo-gx-min', s(min(gx))); val('mo-gx-max', s(max(gx)))
    pr = st2[(st2.variant == 'global') & (st2.learner == 'additive') & (st2.contrast == 'ID-R')]
    lo_h = pr.lo_pp / 100 * pr['T']; hi_h = pr.hi_pp / 100 * pr['T']
    val('mo-idra-gain', str(int(((pr.d_hits >= np.maximum(5, .1 * pr.ceiling_ref)) & (lo_h > 0)).sum())))
    val('mo-idra-excl', str(int(((pr.ceiling_ref >= 50) & (hi_h < .1 * pr.ceiling_ref)).sum())))
    val('mo-idra-elig', str(int((pr.ceiling_ref >= 50).sum())))
    h = pd.read_csv(IU / '12_hgb_id_minus_r.csv')
    for v in ('global', 'family'):
        x = h[h.variant == v]
        val(f'mo-idrh-{v}-gain', str(int(x.meaningful_gain.sum()))); val(f'mo-idrh-{v}-excl', str(int(x.excluded.sum())))
        val(f'mo-idrh-{v}-elig', str(int((x.ceiling_ref >= 50).sum())))
    allc = st2[st2.contrast == 'ID-R']
    val('mo-idr-all-pos', str(int((allc.d_hits > 0).sum()))); val('mo-idr-all-zero', str(int((allc.d_hits == 0).sum())))
    val('mo-idr-all-neg', str(int((allc.d_hits < 0).sum()))); val('mo-idr-all-n', str(len(allc)))
    gxa = ff[(ff.new == 'GX_large') & (ff.ref == 'G1_add')]
    val('mo-gx-llworse', str(int((gxa.drop_duplicates(['inst', 'year']).logloss_new > gxa.drop_duplicates(['inst', 'year']).logloss_ref).sum())))
    val('mo-gx-excl', str(int(gxa.excluded.sum()))); val('mo-gx-elig', str(int((gxa.ceiling_ref >= 50).sum())))
    mortgage_supplement(st2, ff)


def mortgage_supplement(st2, ff):
    # Step 2 all contrasts
    rows = []
    for _, r in st2.sort_values(['variant', 'learner', 'contrast', 'inst', 'year', 'target']).iterrows():
        rows.append([r.variant, 'HGB' if r.learner == 'interaction' else 'additive', r.contrast.replace('-', '$-$'), f'{CNAME[r.inst]} {r.year}', r.target,
                     f'{int(r.hits_ref):,}', s(r.d_hits), ci(r.lo_pp / 100 * r['T'], r.hi_pp / 100 * r['T']), 'Bonf.-12' if r.interval == 'bonferroni12' else '95\\%'])
    write('s_mo_step2', tab(['Set', 'Learner', 'Contrast', 'Cohort', 'Target', 'Ref.\\ hits', r'$\Delta H$', 'Interval', 'Type'], rows, 'lllllrrll',
                            caption='Mortgage representation comparisons (R, SYM, ID), all configurations. Intervals in hits.', label='tab:s_mo_step2', longtable=True, size='\\scriptsize'))
    st = pd.read_csv(IU / '03_posthoc_linear_stacking.csv')
    rows = [[r.variant, f'{CNAME[r.inst]} {r.year}', r.target, f'{int(r.hits_R):,}', f'{int(r.hits_ID):,}', s(r.d_hits_ID_R),
             ci(r.lo95_pp / 100 * r['T'], r.hi95_pp / 100 * r['T']), f'{r.d_logloss_ID_R:+.5f}'.replace('+', '$+$').replace('-', '$-$')] for _, r in st.iterrows()]
    write('s_mo_stacking', tab(['Set', 'Cohort', 'Target', 'R hits', 'ID hits', r'$\Delta H$', '95\\%', r'$\Delta$ log loss'], rows, 'lllrrrll',
                               caption='Post-hoc linear stacking: ID$-$R with standardized logits and $L_2$ logistic regression.', label='tab:s_mo_stacking', longtable=True, size='\\scriptsize'))
    tu = pd.read_csv(IU / '04_toure_formal.csv')
    tu = tu[tu.new.isin(['G2T', 'G2Ttot'])]
    rows = [[r.variant, 'HGB' if r.learner == 'interaction' else 'additive', r.new.replace('G2Ttot', 'G1$+C_0+C_1$').replace('G2T', 'G1$+C_1$'),
             f'{CNAME[r.inst]} {r.year}', r.target, f'{int(r.hits_ref):,}', s(r.d_hits), ci(r.lo_hits, r.hi_hits), 'Bonf.-12' if r.interval.startswith('bonf') else '95\\%']
            for _, r in tu.iterrows()]
    write('s_mo_toure', tab(['Set', 'Learner', 'Representation', 'Cohort', 'Target', 'G1 hits', r'$\Delta H$', 'Interval', 'Type'], rows, 'llllrrrll',
                            caption='Per-class epistemic contribution $C_k=\\mathrm{Var}(p_k)/(2\\mu_k)$ \\citep{toure2026per} added to G1.', label='tab:s_mo_toure', longtable=True, size='\\scriptsize'))
    b2 = pd.read_csv(IU / '05b_full_feature_dropped_unseen.csv')
    rows = []
    for tag, d in (('all', ff), ('dropped', b2)):
        for _, r in d.iterrows():
            rows.append([tag, r.new.replace('_', '\\_'), r.ref.replace('_', '\\_'), f'{CNAME[r.inst]} {r.year}', r.target, f'{int(r.hits_ref):,}', s(r.d_hits),
                         ci(r.lo_hits, r.hi_hits), f'{r.logloss_ref:.4f}', f'{r.logloss_new:.4f}'])
    write('s_mo_fullfeature', tab(['Fields', 'New', 'Reference', 'Cohort', 'Target', 'Ref.\\ hits', r'$\Delta H$', '95\\%', 'LL ref.', 'LL new'], rows, 'llllrrrlrr',
                                  caption="Full-feature incremental comparison (HGB on $S$, $C$ and origination fields). ``dropped'' omits fields whose later-vintage levels were mostly unseen. LL: evaluation log loss. Not an oracle bound.", label='tab:s_mo_fullfeature', longtable=True, size='\\scriptsize', sep='2pt'))
    cv = pd.read_csv(IU / '06_consensus_variants.csv')
    rows = [[r.variant, r.new.replace('_', '\\_'), r.ref.replace('_', '\\_'), f'{CNAME[r.inst]} {r.year}', r.target, f'{int(r.hits_ref):,}', s(r.d_hits), ci(r.lo_hits, r.hi_hits)]
            for _, r in cv.iterrows()]
    write('s_mo_consensus', tab(['Set', 'New', 'Reference', 'Cohort', 'Target', 'Ref.\\ hits', r'$\Delta H$', '95\\%'], rows, 'llllrrrl',
                                caption='Consensus definitions (mean, logit-mean, median, family-balanced), additive learner.', label='tab:s_mo_consensus', longtable=True, size='\\scriptsize'))
    ub = pd.read_csv(IU / '07_unknown_outcome_bounds.csv')
    rows = [[r.year, r.contrast.replace('-', '$-$').replace('AboveTauLow', 'AboveTauLow'), r.target, f'{int(r.exchanged):,}', s(r.legacy_dH), s(r.known_d),
             f'{int(r.u_plus)} / {int(r.u_minus)}', ci(r.bound_lo, r.bound_hi), f'{r.materiality:.1f}'] for _, r in ub.iterrows()]
    write('tab_strict', tab(['Year', 'Contrast', 'Target', 'Exchanged', r'$\Delta H$ recorded', '$d$ (known)', '$u_+$ / $u_-$', 'Bound on $\\Delta H_{\\rm strict}$', 'Threshold'],
                            rows, 'rllrrrrlr'))
    val('mo-strict-umin', str(int((ub.u_plus + ub.u_minus).min()))); val('mo-strict-umax', str(int((ub.u_plus + ub.u_minus).max())))
    val('mo-strict-widthmin', str(int((ub.bound_hi - ub.bound_lo).min()))); val('mo-strict-widthmax', str(int((ub.bound_hi - ub.bound_lo).max())))
    val('mo-strict-n', str(len(ub)))
    qs = pd.read_csv(IU / '10_queue_stability.csv')
    for inst in ('freddie_mac', 'fannie_mae'):
        for pol in ('AllErr_G1', 'AllErr_G2', 'FN_G1', 'FN_G2', 'Range'):
            x = qs[(qs.inst == inst) & (qs.policy == pol)].pairwise_J_mean
            val(f'mo-stab-{inst}-{pol}', f'{x.min():.3f}--{x.max():.3f}')
    rows = [[f'{CNAME[r.inst]} {r.year}', r.policy.replace('_', ' '), f'{r.pairwise_J_mean:.3f}', f'{r.pairwise_J_min:.3f}', f'{r.vs_original_J_mean:.3f}'] for _, r in qs.iterrows()]
    write('s_mo_stability', tab(['Cohort', 'Queue', 'Mean pairwise $J$', 'Min pairwise $J$', 'Mean $J$ vs original'], rows, 'llrrr',
                                caption='Queue membership stability across the three v1.9 candidate reconstructions (10\\% budget).', label='tab:s_mo_stability', longtable=True, size='\\scriptsize'))
    oa = pd.read_csv(ROOT / 'docs/revisions/kbs_v1_9/outcome_audit.csv'); oa = oa[oa.scope == 'historical_test']
    rows = [[r.cohort.split('_')[-1], f'{int(r.N):,}', f'{int(r.legacy_positive):,}', f'{int(r.calendar_event_positive):,}', f'{int(r.strict_known):,}',
             f'{int(r.strict_unknown_excluded):,}'] for _, r in oa.iterrows()]
    write('s_mo_outcome', tab(['Vintage', '$N$', 'Recorded positives', 'Calendar positives', 'Strict-known $N$', 'Unknown'], rows, 'lrrrrr'))
    uc = pd.read_csv(IU / '12_unseen_category_share.csv'); uc = uc[uc.share_unseen > 0]
    rows = [[f'{CNAME[r.inst]} {r.year}', r.feature.replace('_', '\\_'), int(r.levels_2018), 'native' if r.native_categorical else 'numeric/ordinal',
             f(100 * r.share_unseen, 2, True)] for _, r in uc.iterrows()]
    write('s_mo_unseen', tab(['Cohort', 'Field', '2018 levels', 'Encoding', 'Unseen rows'], rows, 'llrlr'))
    rs = pd.read_csv(IU / '12_rebuild_same_seed_hits.csv')
    rs = rs[rs.cohort.str.contains('2021|2022|2023')]
    val('mo-seed-allerr', f"{s(rs[rs.target == 'AllErr'].hits_difference.min())} to {s(rs[rs.target == 'AllErr'].hits_difference.max())}")
    val('mo-seed-fn', f"{s(rs[rs.target == 'FN'].hits_difference.min())} to {s(rs[rs.target == 'FN'].hits_difference.max())}")
    rows = [[r.cohort.replace('_', ' '), r.target, r.seed, int(r.M), int(r.G1_hits), int(r.G2_hits), s(r.hits_difference), f'{r.queue_jaccard:.3f}'] for _, r in rs.iterrows()]
    write('s_mo_sameseed', tab(['Cohort', 'Target', 'Seed', '$M$', 'G1 hits', 'G2 hits', r'$\Delta H$', '$J$(G1, G2)'], rows, 'llrrrrrr',
                               caption='G2$-$G1 within each candidate reconstruction (from the v1.9 reconstruction outputs).', label='tab:s_mo_sameseed', longtable=True, size='\\scriptsize'))


# ---------------------------------------------------------------- discovery
DNAME = {'uci': 'UCI', 'fico': 'FICO', 'hmda': 'HMDA', 'lendingclub': 'Lending Club', 'gmc': 'GMC', 'hmeq': 'HMEQ', 'home_credit': 'Home Credit'}


def discovery():
    P = json.loads((ROOT / 'idea-update/protocol_audit_discovery_v1.json').read_text())
    roles = pd.read_csv(IU / '13_discovery_roles.csv').set_index('dataset')
    m = pd.read_csv(IU / '14_methods.csv'); c = pd.read_csv(IU / '14_contrasts.csv'); run = json.loads((IU / '14_run_info.json').read_text())
    val('disc-frozen', P['frozen_at'].replace('T', ' ')[:16] + ' AEDT')
    import hashlib
    val('disc-sha-protocol', hashlib.sha256((ROOT / 'idea-update/protocol_audit_discovery_v1.json').read_bytes()).hexdigest()[:12])
    val('disc-sha-script', hashlib.sha256((ROOT / 'idea-update/scripts/14_discovery_eval.py').read_bytes()).hexdigest()[:12])
    val('disc-sha-roles', P['data_roles']['sha256'][:12])
    val('disc-draws', f"{run['draws']:,}")
    pr = c[c.role == 'primary']; se = c[c.role == 'secondary']
    val('disc-n-primary', str(len(pr))); val('disc-n-secondary', str(len(se)))
    val('disc-primary-gains', str(int(pr.meaningful_gain.sum()))); val('disc-primary-losses', str(int(pr.meaningful_loss.sum())))
    val('disc-secondary-gains', str(int(se.meaningful_gain.sum()))); val('disc-secondary-losses', str(int(se.meaningful_loss.sum())))
    rows = []
    for d in P['datasets']:
        r = roles.loc[d]; mm = m[(m.dataset == d) & (m.target == 'AllErr')].iloc[0]
        redundant = r.M_global == 2 and r.reference_in_global
        cells, concl = [], []
        for con in ('ID-SYM', 'ID-G1'):
            x = pr[(pr.dataset == d) & (pr.contrast == con)].iloc[0]
            key = f'disc-{d}-{con.lower().replace("-", "")}'
            val(key + '-dh', s(x.d_hits)); val(key + '-ci', ci(x.lo_adjusted, x.hi_adjusted)); val(key + '-thr', f'{x.materiality_threshold:.1f}')
            val(key + '-r', f'{int(x.remaining_capacity_ref):,}'); val(key + '-hi', s(x.hi_adjusted))
            val(key + '-per10k', s(x.d_hits_per10k, 1)); val(key + '-precpp', s(x.d_precision_pp, 1))
            cells.append(f'{s(x.d_hits)} {ci(x.lo_adjusted, x.hi_adjusted)}')
            if redundant:
                concl.append('eq')
            elif x.meaningful_gain:
                concl.append('gain')
            elif x.excluded:
                concl.append('bounded')
            else:
                concl.append('uncertain')
        thr = [f"{pr[(pr.dataset == d) & (pr.contrast == k)].iloc[0].materiality_threshold:.1f}" for k in ('ID-SYM', 'ID-G1')]
        if redundant:
            text = r'Information-equivalent control ($P_2=2C-S$)'
        elif concl == ['bounded', 'bounded']:
            text = r'Bounded: both upper limits below thresholds'
        elif all(k == 'uncertain' for k in concl):
            text = r'Uncertain: intervals include gains $\geq$ thresholds'
        else:
            text = '; '.join(concl)
        mlab = f"{int(r.M_global)}" + (r'\textsuperscript{\dag}' if redundant else '')
        rows.append([DNAME[d], f'{int(mm.N):,}', f'{int(mm["T"]):,}', f'{int(mm.B):,}', mlab, cells[0], cells[1], f'{thr[0]} / {thr[1]}', text])
    write('tab_discovery', tab(['Dataset', '$N$', 'AllErr', '$B$', '$M$', r'ID$-$SYM [Bonf.-14]', r'ID$-$G1 [Bonf.-14]', 'Thresholds', 'Conclusion'],
                               rows, 'lrrrrlll>{\\raggedright\\arraybackslash}p{3.3cm}'))
    # info-equivalent controls: range of representation differences relative to G1 (AllErr)
    for d in ('hmda', 'home_credit'):
        x = c[(c.dataset == d) & (c.target == 'AllErr') & c.contrast.isin(['G2-G1', 'SYM-G1', 'ID-G1'])]
        val(f'disc-{d}-eqmin', s(x.d_hits.min())); val(f'disc-{d}-eqmax', s(x.d_hits.max()))
    # log-loss: unique (dataset, new, ref) pairs
    u = c.drop_duplicates(['dataset', 'contrast'])
    val('disc-ll-pairs', str(len(u))); val('disc-ll-worse', str(int((u.d_logloss_lo > 0).sum()))); val('disc-ll-better', str(int((u.d_logloss_hi < 0).sum())))
    better = u[u.d_logloss_hi < 0]
    val('disc-ll-better-list', '; '.join(f"{DNAME[r.dataset]} {r.contrast.replace('-', '$-$')}" for _, r in better.iterrows()))
    gx = u[(u.dataset == 'gmc') & (u.contrast == 'GX-G1')]
    if len(gx): val('disc-gmc-gx-dh', s(gx.iloc[0].d_hits)); val('disc-gmc-gx-thr', f'{gx.iloc[0].materiality_threshold:.1f}')
    g = u[(u.dataset == 'gmc') & (u.contrast == 'ID-SYM')]
    val('disc-gmc-idsym-ll', f'{g.iloc[0].d_logloss:+.5f}'.replace('+', '$+$').replace('-', '$-$'))
    val('disc-gmc-gx-ll', f'{gx.iloc[0].d_logloss:+.5f}'.replace('+', '$+$').replace('-', '$-$'))
    g1v = c[(c.contrast == 'G1-G1add')]
    val('disc-g1add-gain', str(int(g1v.meaningful_gain.sum()))); val('disc-g1add-loss', str(int(g1v.meaningful_loss.sum())))
    val('disc-g1add-excl-allerr', str(int(g1v[g1v.target == 'AllErr'].excluded.sum())))
    discovery_supplement(P, roles, m, c)
    fig_discovery(P, roles, pr)


def discovery_supplement(P, roles, m, c):
    rows = []
    for d in P['datasets']:
        r = roles.loc[d]
        short = {'uci': 'credit-card default', 'fico': 'HELOC risk performance (bad)', 'hmda': 'loan origination (not repayment)',
                 'lendingclub': 'loan default', 'gmc': '90+ days past due within two years', 'hmeq': 'default or serious delinquency',
                 'home_credit': 'payment difficulties'}[d]
        rows.append([DNAME[d], short, f'{int(r.n_train):,}', f'{int(r.n_val):,}', f'{int(r.n_test):,}', f'{r.tau:.4f}',
                     f'{int(r.M_global)} ({int(r.M_cached_family)})', r.reference_model.replace('_', '\\_\\allowbreak '), 'yes' if r.reference_in_global else 'no'])
    write('s_disc_roles', tab(['Dataset', 'Label', 'Train', 'Validation', 'Test', r'$\tau$', '$M$ global (cached)', 'Reference model', 'In set'], rows,
                              'l>{\\raggedright\\arraybackslash}p{2.6cm}rrrrr>{\\raggedright\\arraybackslash}p{1.6cm}>{\\raggedright\\arraybackslash}p{2.1cm}r', caption='Data roles for the seven discovery datasets. $\\tau$ is the validation prevalence; base models were trained on the training split and screened on validation AUC.',
                              label='tab:s_disc_roles', longtable=True, size='\\scriptsize', sep='2pt'))
    rows = []
    for _, r in m.iterrows():
        rows.append([DNAME[r.dataset], r.target, r.method.replace('G1add', 'G1 additive'), f'{int(r.hits):,}', f'{int(r["T"]):,}', f'{100 * r.capture:.1f}', f'{100 * r.precision:.1f}', f'{int(r.fold_capacity):,}'])
    write('s_disc_methods', tab(['Dataset', 'Target', 'Method', 'Hits', '$T$', 'Capture (\\%)', 'Precision (\\%)', r'$\sum_k\min(B_k,T_k)$'], rows, 'lllrrrrr',
                                caption='All methods, both targets, 10\\% budget (fold-wise queues).', label='tab:s_disc_methods', longtable=True, size='\\scriptsize'))
    rows = []
    for _, r in c.iterrows():
        if not r.defined: continue
        adj = ci(r.lo_adjusted, r.hi_adjusted) if r.role == 'primary' else '--'
        rows.append([DNAME[r.dataset], r.target, r.contrast.replace('-', '$-$').replace('G1add', 'G1$_{\\rm add}$'), r.role, f'{int(r.hits_ref):,}', s(r.d_hits),
                     f'{int(r.remaining_capacity_ref):,}', ci(r.lo_pointwise, r.hi_pointwise), adj, f'{r.materiality_threshold:.1f}',
                     'gain' if r.meaningful_gain else ('loss' if r.meaningful_loss else ('excl.' if r.excluded else '--')),
                     f'{r.d_logloss:+.5f}'.replace('+', '$+$').replace('-', '$-$')])
    write('s_disc_contrasts', tab(['Dataset', 'Target', 'Contrast', 'Role', 'Ref.\\ hits', r'$\Delta H$', 'Rem.', '95\\%', 'Bonf.-14', 'Thr.', 'Flag', r'$\Delta$ log loss'],
                                  rows, 'lllllrrllrll', caption='All 154 contrast rows (14 primary, 140 secondary). Secondary rows are exploratory and use pointwise intervals. The log-loss difference does not depend on the target.',
                                  label='tab:s_disc_contrasts', longtable=True, size='\\scriptsize'))


def fig_discovery(P, roles, pr):
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ds = P['datasets']; y = 0; ticks, labels = [], []
    for d in ds:
        red = roles.loc[d].M_global == 2 and roles.loc[d].reference_in_global
        for j, (con, col) in enumerate((('ID-SYM', BLUE), ('ID-G1', ORANGE))):
            x = pr[(pr.dataset == d) & (pr.contrast == con)].iloc[0]
            t = x.materiality_threshold
            cc = GRAY if red else col
            ax.plot([x.lo_adjusted / t, x.hi_adjusted / t], [y, y], color=cc, lw=1.6, solid_capstyle='round')
            ax.plot(x.d_hits / t, y, 'o', ms=5.5, color=cc, mec='white', mew=1,
                    label=None if d != 'uci' else ('ID$-$SYM' if con == 'ID-SYM' else 'ID$-$G1'))
            y += 1
        ticks.append(y - 1.5); labels.append(DNAME[d] + ('$^\\dagger$' if red else ''))
        y += .6
    ax.axvline(0, color=INK2, lw=.8); ax.axvline(1, color=GRAY, lw=1, ls=(0, (1, 2))); ax.axvline(-1, color=GRAY, lw=1, ls=(0, (1, 2)))
    ax.text(1, 1.01, 'threshold', transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, ha='center', va='bottom')
    ax.text(-1, 1.01, '$-$threshold', transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, ha='center', va='bottom')
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel(r'$\Delta H$ / dataset-specific threshold  (Bonferroni-14 intervals)', fontsize=8)
    ax.grid(axis='x', color=GRID, lw=.6); ax.set_axisbelow(True); ax.tick_params(labelsize=7.5, colors=INK2)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.legend(loc='lower right', fontsize=7.5, frameon=False)
    fig.tight_layout(); save(fig, 'fig_discovery')


def main():
    GEN.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK})
    ex = load_exact()
    dv_eps, dv_b = theory(ex); fig_theory(dv_eps, dv_b)
    positive_control(); construction_e(); mortgage(); discovery()
    with open(GEN / 'numbers.tex', 'w') as f:
        f.write('% Generated by src/render_revision_v1_11.py from completed result files; do not edit.\n')
        for k, v in sorted(VALS.items()):
            f.write(f'\\expandafter\\def\\csname val@{k}\\endcsname{{{v}}}\n')
    (GEN / 'numbers.json').write_text(json.dumps(VALS, indent=1, ensure_ascii=False))
    print(f'{len(VALS)} values; tables in {GEN}; figures in {FIG}')


if __name__ == '__main__':
    main()
