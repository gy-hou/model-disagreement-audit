"""Presentation-only figure revision from archived, unchanged KBS results.

Python/matplotlib is the exclusive rendering backend. This adapts the existing
presentation builders without changing numeric transformations or exclusions.
No fitted models, queue construction, bootstrap draws or scientific simulations
are run. Final exports are .pdf and .svg vectors plus .png previews. The local
Nature QA utilities retain their upstream Apache-2.0 license.
"""
from pathlib import Path
import importlib.util
import json
import hashlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import FancyBboxPatch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'docs/revisions/kbs_final_r1'
FIG = ROOT / 'paper/kbs_final_r1/figures'
QA = OUT / 'figure_qa'
VALUES = OUT / 'figure_values'
for folder in [FIG, QA, VALUES]:
    folder.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(OUT / 'nature_qa'))
from audit_panel_alignment import require_matplotlib_panel_alignment

INK, MUTED, GRID = '#22313D', '#5D6872', '#E1E6EA'
BLUE, ORANGE, GREEN, PURPLE = '#2C6E9F', '#C16C35', '#348B76', '#9D7399'
STEMS = ['protocol_v1_10', 'historical_targets_v1_10', 'score_geometry_v1_10',
         'final_mechanisms', 'incremental_capture_v1_10', 'frozen_selection_v1_10',
         'candidate_stability_v1_10']
CHECKS, SOURCE_HASHES = {}, {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    SOURCE_HASHES[str(path.relative_to(ROOT))] = sha(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def configure():
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9,
        'axes.labelsize': 9, 'axes.titlesize': 10, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'legend.fontsize': 9, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'kbs-final-r1', 'axes.titlecolor': INK,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.linewidth': .7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False})


def geometry_snapshot(fig):
    """Numeric artist coordinates are unchanged by the presentation pass."""
    refs = []
    for ax in fig.axes:
        for line in ax.lines:
            refs.append((line, 'line', np.asarray(line.get_xydata()).copy()))
        for col in ax.collections:
            refs.append((col, 'offsets', np.asarray(col.get_offsets()).copy()))
            refs.append((col, 'paths', [p.vertices.copy() for p in col.get_paths()]))
            if hasattr(col, 'get_segments'):
                refs.append((col, 'segments', [np.asarray(x).copy() for x in col.get_segments()]))
        for patch in ax.patches:
            if isinstance(patch, plt.Rectangle):
                refs.append((patch, 'rectangle', np.array([*patch.get_xy(), patch.get_width(), patch.get_height()])))
    return refs


def verify_geometry(refs):
    for obj, kind, before in refs:
        if kind == 'line':
            after = obj.get_xydata()
        elif kind == 'offsets':
            after = obj.get_offsets()
        elif kind == 'rectangle':
            after = np.array([*obj.get_xy(), obj.get_width(), obj.get_height()])
        elif kind == 'paths':
            after = [p.vertices for p in obj.get_paths()]
        else:
            after = obj.get_segments()
        if kind in ['segments', 'paths']:
            assert len(before) == len(after)
            assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(before, after))
        else:
            assert np.array_equal(before, np.asarray(after), equal_nan=True)


def polish(fig, stem):
    for text in fig.findobj(match=Text):
        text.set_fontfamily('Arial')
        text.set_color(INK if text.get_color() in ['black', '.15', '#203344'] else text.get_color())
    for ax in fig.axes:
        ax.tick_params(axis='both', labelsize=9, colors=MUTED, width=.65, length=3)
        ax.spines['bottom'].set_color('#A7B1B9')
        ax.spines['bottom'].set_linewidth(.65)
        for grid in ax.get_xgridlines()+ax.get_ygridlines():
            grid.set_color(GRID)
            grid.set_linewidth(.55)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
        for title in [ax.title, ax._left_title, ax._right_title]:
            title.set_fontfamily('Arial')
            title.set_fontweight('bold')
    if stem == 'historical_targets_v1_10':
        for ax in fig.axes:
            # Range is the candidate-disagreement strategy in every panel.
            ax.axhspan(.68, 1.32, xmax=.82, color='#F8F0E9', zorder=-4, lw=0)
            ax.add_patch(plt.Rectangle((.84, -.02), .20, 1.04,
                transform=ax.transAxes, color='#F6F8FA', zorder=-5, lw=0, clip_on=False))
            ax.set_xlabel('Capture of target (%)')
            ax.set_xticks([0, 20, 40])
    elif stem == 'score_geometry_v1_10':
        for ax in fig.axes:
            if ax.get_xlabel() == 'Deployment score':
                ax.set_xlabel('Reference score')
        # Modality rows keep their scientific order and the original data.
        for ax in fig.axes:
            if 'Signed decomposition' in ax.get_title(loc='left'):
                ax.axhspan(.65, 1.35, color='#F5F7F9', zorder=-5, lw=0)
    elif stem == 'incremental_capture_v1_10':
        for text in fig.texts:
            if text.get_text().startswith('G1: deployment'):
                text.set_text('G1: reference score + consensus    G2: G1 + range')
        for ax in fig.axes:
            for y in [1, 3, 5]:
                ax.axhspan(y-.42, y+.42, color='#F6F8FA', zorder=-5, lw=0)
            ax.set_xlabel('Capture difference (percentage points)')
    elif stem == 'frozen_selection_v1_10':
        table = fig.axes[0]
        table.plot([0, .99], [4.39, 4.39], color='#A7B1B9', lw=.65)
        for text in table.texts:
            if text.get_text() in ['Base choice', 'Full choice', 'OOF hits', 'Institution / target']:
                text.set_fontweight('bold')
    elif stem == 'candidate_stability_v1_10':
        for ax in fig.axes:
            ax.axhspan(.65, 1.35, color='#F6F8FA', zorder=-5, lw=0)
    elif stem == 'final_mechanisms':
        a, b, c, d = fig.axes
        # Replace labels crossing reference lines with a clear external summary.
        for text in list(c.texts):
            if text.get_text().startswith('Mean '):
                text.remove()
        c.set_title('C  E: screening + refitting', loc='left', weight='bold', pad=26)
        c.text(0, 1.035, '80k labels: +0.37 expected hits / 10,000',
               transform=c.transAxes, fontsize=8.5, color=BLUE, va='bottom')
        c.set_ylabel('Training labels')
        b.set_title('B  E: budget boundary', loc='left', weight='bold', pad=15)
        for text in list(b.texts):
            if text.get_text() == 'Same queue':
                text.remove()
        b.text(.97, -.27, '20%: same oracle queue', transform=b.transAxes,
               ha='right', fontsize=8.5, color=MUTED)
        d.set_title('D  Fixed-design comparison', loc='left', weight='bold', pad=26)
        d.text(0, 1.035, 'Oracle and fitted results', transform=d.transAxes,
               fontsize=8.5, color=MUTED, va='bottom')
        for ax in [c, d]:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', length=0)


def bounded_vector_hatches(fig):
    """Preserve hatch encodings without off-panel PDF pattern tile geometry."""
    fig.canvas.draw()
    for patch in fig.findobj():
        if not hasattr(patch, 'get_hatch') or not patch.get_hatch():
            continue
        hatch = patch.get_hatch()
        bounds = patch.get_path().transformed(patch.get_transform()).get_extents()
        patch.set_hatch(None)
        x0, y0, x1, y1 = [float(v)*72/fig.dpi for v in bounds.extents]
        width, height = x1-x0, y1-y0
        if width <= 0 or height <= 0:
            continue
        for diagonal in ([1, -1] if 'x' in hatch else [1]):
            for offset in np.arange(-height, width, 4.5):
                low, high = max(0, -offset), min(height, width-offset)
                if high <= low:
                    continue
                xs = [x0+offset+low, x0+offset+high]
                ys = ([y0+low, y0+high] if diagonal == 1 else [y1-low, y1-high])
                fig.add_artist(Line2D(np.array(xs)/(fig.get_figwidth()*72),
                    np.array(ys)/(fig.get_figheight()*72), transform=fig.transFigure,
                    color='white', lw=.45, solid_capstyle='butt', clip_on=False, zorder=10))


def save_figure(fig, stem):
    refs = geometry_snapshot(fig)
    polish(fig, stem)
    verify_geometry(refs)
    if stem == 'score_geometry_v1_10':
        bounded_vector_hatches(fig)
    fig.canvas.draw()
    options = {}
    if stem == 'incremental_capture_v1_10':
        options = {'panel_ids': ['A', 'B'], 'column_groups': [['A', 'B']]}
    elif stem == 'frozen_selection_v1_10':
        # A is a text table, not a comparable numeric plotting region.
        options = {'exclude_axes': [fig.axes[0]]}
    elif stem == 'score_geometry_v1_10':
        # Original row heights are intentionally unequal for different roles.
        options = {'exemptions': [{'panels': ['a', 'b', 'c', 'd', 'e', 'f'],
            'checks': ['vertical-gutter'],
            'reason': 'Different modality rows reserve space for a shared composition legend.'}]}
    alignment = require_matplotlib_panel_alignment(fig,
        json_out=QA/f'{stem}.alignment.json', overlay_svg=QA/f'{stem}.alignment.svg',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, **options)
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        if box.x0 < -.001 or box.y0 < -.001 or box.x1 > 1.001 or box.y1 > 1.001:
            outside.append(text.get_text())
    assert not outside, (stem, outside)
    outputs = {}
    for ext in ['pdf', 'svg', 'png']:
        path = FIG/f'{stem}.{ext}'
        metadata = ({'CreationDate': None, 'ModDate': None} if ext == 'pdf'
                    else {'Date': None} if ext == 'svg' else None)
        fig.savefig(path, dpi=300, facecolor='white', metadata=metadata)
        outputs[str(path.relative_to(ROOT))] = sha(path)
    CHECKS[stem] = dict(outputs=outputs, numeric_artist_geometry_unchanged=True,
        visible_text_inside_canvas=True, alignment_verdict=alignment['verdict'],
        size_inches=list(map(float, fig.get_size_inches())))
    plt.close(fig)


def protocol():
    """Same six roles and counts, with a distinct question/data hierarchy."""
    path = ROOT/'results/tables/v1_10_presentation/figure1_data_roles.csv'
    SOURCE_HASHES[str(path.relative_to(ROOT))] = sha(path)
    roles = pd.read_csv(path)
    assert list(roles.cases_per_institution.iloc[:3]) == [30000, 10000, 10000]
    assert len(roles) == 4 and pd.isna(roles.cases_per_institution.iloc[3])
    roles.to_csv(VALUES/'figure1_data_roles.csv', index=False, float_format='%.17g')
    fig = plt.figure(figsize=(6.3, 3.5))
    ax = fig.add_axes([.025, .015, .95, .97])
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    ax.text(0, .975, 'THREE QUESTIONS', color=MUTED, weight='bold', fontsize=8.5)
    xs = [.0, .345, .69]
    top = [('1', 'Different cases', 'Target and budget\nScore region\nQueue composition'),
           ('2', 'Additional information', 'Conditional information\nBudget boundary\nLearner approximation'),
           ('3', 'Extra target hits', 'Matched G0 / G1 / G2\nFrozen policy choice\nCandidate reconstruction')]
    for x, (num, title, body), color in zip(xs, top, [BLUE, GREEN, ORANGE]):
        ax.add_patch(FancyBboxPatch((x, .635), .30, .285, boxstyle='round,pad=.01,rounding_size=.012',
                    facecolor='#F6F8FA', edgecolor='#D9E0E5', lw=.65))
        ax.text(x+.012, .878, num, color=color, weight='bold', fontsize=14)
        ax.text(x+.047, .880, title, color=INK, weight='bold', fontsize=9.1)
        ax.text(x+.012, .804, body, va='top', fontsize=8.9, linespacing=1.4)
    ax.text(0, .555, 'MORTGAGE DATA ROLES', color=MUTED, weight='bold', fontsize=8.5)
    ax.text(.99, .555, '2018 counts per institution', ha='right', color=MUTED, fontsize=8.5)
    bottom = [('2018 credit models', 'Train: 30,000\nValidation: 10,000\nModels, candidates, τ'),
              ('2018 audit models', 'Former test: 10,000\nFive-fold choice\nFull-data refit'),
              ('2021–2023 evaluation', 'Six historical cohorts\nFixed models / policies\nTarget hits / capture')]
    for x, (title, body) in zip(xs, bottom):
        ax.add_patch(FancyBboxPatch((x, .16), .30, .32, boxstyle='round,pad=.01,rounding_size=.012',
                    facecolor='white', edgecolor='#D9E0E5', lw=.65))
        ax.text(x+.012, .428, title, weight='bold', fontsize=9.0)
        ax.text(x+.012, .350, body, va='top', fontsize=9, linespacing=1.45)
    for y in [.777, .32]:
        for x in [.312, .657]:
            ax.annotate('', xy=(x+.025, y), xytext=(x, y),
                        arrowprops={'arrowstyle': '->', 'color': MUTED, 'lw': .8})
    ax.text(.5, .048, 'Same review budget; different questions require different targets.',
            ha='center', color=MUTED, fontsize=9)
    save_figure(fig, 'protocol_v1_10')


def montage():
    canvas = Image.new('RGB', (1400, 2200), '#e7e7e7')
    draw = ImageDraw.Draw(canvas)
    for i, stem in enumerate(STEMS):
        im = Image.open(FIG/f'{stem}.png').convert('RGB')
        im.thumbnail((670, 510))
        x, y = 15+(i % 2)*700, 20+(i//2)*550
        canvas.paste(im, (x, y+22)); draw.text((x, y), f'Figure {i+1}: {stem}', fill='black')
    canvas.save(OUT/'figures_after_montage.png')


def main():
    configure()
    base = module(ROOT/'src/render_revision_v1_10.py', 'kbs_base_figures')
    base.OUT = VALUES
    base.FIG = FIG
    base.BLUE, base.ORANGE, base.GREEN, base.PURPLE = BLUE, ORANGE, GREEN, PURPLE
    base.INK, base.MUTED, base.GRID = INK, MUTED, GRID
    base.save = save_figure
    protocol()
    historical = base.historical()
    base.geometry(historical)
    joined = base.main_comparisons()
    base.incremental(joined)
    base.selection(joined)
    base.stability()
    # The construction builder already checks the complete archived results.
    con = module(ROOT/'docs/revisions/kbs_final/final_construction_figure.py', 'kbs_construction')
    con.OUT = VALUES
    con.FIG = FIG
    con.BLUE, con.ORANGE, con.GREEN = BLUE, ORANGE, GREEN
    data = con.verify_and_summarize()
    captured = []
    original_save = Figure.savefig
    original_close = plt.close
    def capture(self, *args, **kwargs):
        if not captured:
            captured.append(self)
    Figure.savefig = capture
    plt.close = lambda *args, **kwargs: None
    try:
        con.render(*data)
    finally:
        Figure.savefig = original_save
        plt.close = original_close
    assert len(captured) == 1
    configure()
    save_figure(captured[0], 'final_mechanisms')
    SOURCE_HASHES.update(base.SOURCES)
    SOURCE_HASHES.update(con.SOURCES)
    assert all(sha(ROOT/p) == h for p, h in SOURCE_HASHES.items())
    equal = {}
    for path in VALUES.glob('*.csv'):
        old = (ROOT/'docs/revisions/kbs_final'/path.name if path.name.startswith('final_')
               else ROOT/'results/tables/v1_10_presentation'/path.name)
        assert old.exists(), old
        a, b = pd.read_csv(path), pd.read_csv(old)
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        equal[path.name] = {'rows': len(a), 'columns': list(a.columns), 'all_cells_equal': True}
    montage()
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        scientific_rerun=False, backend='python/matplotlib', fonts='Arial',
        manuscript_text_width_mm=160, figure_source_width_mm=160.02,
        source_sha256=SOURCE_HASHES, data_equality=equal, figures=CHECKS,
        script_sha256=sha(__file__))
    (OUT/'FIGURE_VALIDATION.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'figures': list(CHECKS), 'numeric_tables_equal': len(equal)}, indent=2))


if __name__ == '__main__':
    main()
