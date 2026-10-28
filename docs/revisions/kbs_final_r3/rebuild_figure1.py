"""Rebuild Figure 1 (study design) for final r3.

Presentation only. The three per-institution counts are read from the archived
figure1_data_roles.csv used by the r1 builder; every other label restates the
manuscript (Sections 4.1, 5.1-5.5 and 6.3-6.6). No model, simulation or
bootstrap is run. Outputs go to paper/kbs_final_r3/figures and are copied
byte-for-byte to paper/kbs_final_r3_zh/figures; checks go to FIGURE1_QA.json.
"""
from pathlib import Path
import hashlib
import itertools
import json
import shutil
import subprocess

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'docs/revisions/kbs_final_r3'
FIG = ROOT / 'paper/kbs_final_r3/figures'
FIG_ZH = ROOT / 'paper/kbs_final_r3_zh/figures'
ROLES = ROOT / 'results/tables/v1_10_presentation/figure1_data_roles.csv'
STEM = 'protocol_v1_10'

INK, MUTED = '#22313D', '#5D6872'
BLUE, ORANGE = '#2C6E9F', '#C16C35'
FILL, EDGE, HEAD, RANGE_BG, RULE = '#F6F8FA', '#D9E0E5', '#EAEFF3', '#F8F0E9', '#C9D2D9'
W, H = 6.3, 4.10          # inches; 6.3 in is the 160 mm text width
XL, XR = 0.05, 6.25       # content edges, inside the canvas
ROUND = 0.06              # corner radius
STEP = 0.19               # body line step

CLAIMS = [('1', 'Queues differ', 'different cases reviewed',
           'Exact counterexample', '1 without 2 · §4.1'),
          ('2', 'Added information', r'beyond $S$ and consensus',
           'Construction E', '2 without 3 at 20% · §6.3'),
          ('3', 'Extra target hits', 'at the declared budget',
           'Matched G2 − G1', 'measures 3 · §6.4–6.6')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure():
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 8.5, 'text.color': INK,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'kbs-final-r3', 'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'})


def counts():
    roles = pd.read_csv(ROLES)
    n = dict(zip(roles.data_role, roles.cases_per_institution))
    assert list(n) == ['2018 training', '2018 validation', '2018 former test',
                       '2021–2023 historical cohorts'], list(n)
    assert [n['2018 training'], n['2018 validation'], n['2018 former test']] == [30000, 10000, 10000]
    assert pd.isna(n['2021–2023 historical cohorts'])
    return {k: f'{int(v):,}' for k, v in n.items() if not pd.isna(v)}


class Canvas:
    """One full-figure axes whose data units are inches."""

    def __init__(self):
        self.fig = plt.figure(figsize=(W, H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, W), ylim=(0, H))
        self.ax.axis('off')
        self.contained = []   # (text, (x0, y0, x1, y1) of its box, inches)

    def text(self, x, y, s, box=None, **kw):
        t = self.ax.text(x, y, s, **kw)
        if box is not None:
            self.contained.append((t, box))
        return t

    def extent(self, artist):
        self.fig.canvas.draw()
        return artist.get_window_extent().transformed(self.ax.transData.inverted())

    def card(self, x, y, w, h, face):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={ROUND}',
                          facecolor=face, edgecolor=EDGE, lw=.7, zorder=1, clip_on=False))

    def header_card(self, x, y, w, h, hh):
        """White card with a tinted header strip of height hh."""
        style = f'round,pad=0,rounding_size={ROUND}'
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor='white',
                          edgecolor='none', zorder=1, clip_on=False))
        self.ax.add_patch(FancyBboxPatch((x, y + h - hh), w, hh, boxstyle=style, facecolor=HEAD,
                          edgecolor='none', zorder=2, clip_on=False))
        self.ax.add_patch(Rectangle((x, y + h - hh), w, ROUND, facecolor=HEAD, edgecolor='none',
                          zorder=2, clip_on=False))
        self.ax.add_line(Line2D([x, x + w], [y + h - hh] * 2, color=EDGE, lw=.7, zorder=3))
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor='none',
                          edgecolor=EDGE, lw=.7, zorder=4, clip_on=False))

    def arrow(self, x0, x1, y):
        self.ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=9,
                          lw=1.0, color=MUTED, shrinkA=0, shrinkB=0, zorder=5, clip_on=False))


def panel_a(c):
    c.text(XL, H - .05, 'A  Three claims and the evidence that separates them',
           fontsize=10, weight='bold', va='top')
    gap, top, height = .34, H - .34, .58
    width = (XR - XL - 2 * gap) / 3
    bottom = top - height
    for i, (num, title, sub, result, note) in enumerate(CLAIMS):
        x = XL + i * (width + gap)
        box = (x, bottom, x + width, top)
        c.card(x, bottom, width, height, FILL)
        c.text(x + .13, top - .19, num, box, color=BLUE, weight='bold', fontsize=11, va='center')
        c.text(x + .34, top - .19, title, box, weight='bold', fontsize=9.5, va='center')
        c.text(x + .34, top - .41, sub, box, color=MUTED, va='center')
        label = (x, bottom - .5, x + width, bottom)
        c.ax.add_line(Line2D([x + .13] * 2, [bottom - .09, bottom - .42], color=RULE, lw=1.4,
                             solid_capstyle='butt'))
        c.text(x + .22, bottom - .17, result, label, weight='bold', va='center')
        c.text(x + .22, bottom - .35, note, label, color=MUTED, fontsize=8, va='center')
        if i < 2:
            c.arrow(x + width + .05, x + width + gap - .05, top - height / 2)
    return bottom


def information_sets(c, x, width, first, box):
    """G0, G1, G2 with the range addition highlighted as Range is in Figures 2-3."""
    rows = [('G0', INK, [(r'logit $S$', INK)]),
            ('G1', INK, [(r'G0 + consensus $C$', INK)]),
            ('G2', ORANGE, [('G1 + ', INK), (r'range $D$', ORANGE)])]
    for j, (name, color, parts) in enumerate(rows):
        y = first - j * STEP
        if name == 'G2':
            c.ax.add_patch(Rectangle((x + .06, y - STEP / 2 + .01), width - .12, STEP - .02,
                                     facecolor=RANGE_BG, edgecolor='none', zorder=2))
        c.text(x + .12, y, name, box, weight='bold', color=color, va='center', zorder=6)
        xx = x + .42
        for s, col in parts:
            xx = c.extent(c.text(xx, y, s, box, color=col, va='center', zorder=6)).x1
    for j, line in enumerate(['Five-fold OOF policy choice', 'Full-data refit']):
        c.text(x + .12, first - (3 + j) * STEP, line, box, va='center')


def panel_b(c, n, bottom_a):
    title_y = bottom_a - .62
    c.text(XL, title_y, 'B  Mortgage protocol, per institution', fontsize=10, weight='bold', va='top')
    gap, hh, height = .30, .40, 1.92
    width = (XR - XL - 2 * gap) / 3
    top = title_y - .28
    bottom = top - height
    stages = [
        ('Candidate construction',
         [f"2018 training: {n['2018 training']}", f"2018 validation: {n['2018 validation']}"],
         [r'Reference score $S$, threshold $\tau$', '26-model candidate grid',
          r'Global AUC gate → set $R$', r'Consensus $C$, range $D$ over $R$']),
        ('Target-aligned audit\ndevelopment',
         [f"2018 former test: {n['2018 former test']}", r'$C_{\rm audit}$, five folds'],
         None),
        ('Fixed-policy evaluation',
         ['2021–2023 former test', 'Six institution–vintage cohorts'],
         ['Models and policy index frozen', 'Targets: all errors, FN',
          'Budget: 10% of each cohort', 'Contrasts: G2 − G1, full − base'])]
    first = top - hh - .17 - 2 * .16 - .11   # centre of the first body line
    for i, (name, data, body) in enumerate(stages):
        x = XL + i * (width + gap)
        box = (x, bottom, x + width, top)
        c.header_card(x, bottom, width, height, hh)
        c.text(x + .12, top - hh / 2, name, box, weight='bold', fontsize=9, va='center',
               linespacing=1.15)
        for j, line in enumerate(data):
            c.text(x + .12, top - hh - .17 - j * .16, line, box, color=MUTED, fontsize=8, va='center')
        if body is None:
            information_sets(c, x, width, first, box)
        else:
            for j, line in enumerate(body):
                c.text(x + .12, first - j * STEP, line, box, va='center')
        if i < 2:
            c.arrow(x + width + .05, x + width + gap - .05, top - hh - (height - hh) / 2)
    c.text(XL, bottom - .17, 'Read with: score geometry (Fig. 3) · remaining hit capacity '
           '(Eq. 6) · candidate reconstructions (Fig. 7)', color=MUTED, fontsize=8, va='center')


def qa(c):
    """Every text inside the canvas and its box; no two texts overlap."""
    c.fig.canvas.draw()
    inv = c.ax.transData.inverted()
    texts = [t for t in c.ax.texts if t.get_text().strip()]
    box = {t: t.get_window_extent().transformed(inv) for t in texts}
    problems = []
    for t, b in box.items():
        if b.x0 < .01 or b.y0 < .01 or b.x1 > W - .01 or b.y1 > H - .01:
            problems.append(['outside canvas', t.get_text()])
    for t, (x0, y0, x1, y1) in c.contained:
        b = box[t]
        if b.x0 < x0 + .04 or b.x1 > x1 - .04 or b.y0 < y0 + .02 or b.y1 > y1 - .02:
            problems.append(['outside its box', t.get_text()])
    for a, b in itertools.combinations(texts, 2):
        A, B = box[a], box[b]
        if min(A.x1, B.x1) - max(A.x0, B.x0) > .005 and min(A.y1, B.y1) - max(A.y0, B.y0) > .005:
            problems.append(['overlap', a.get_text(), b.get_text()])
    return problems, sorted(t.get_text() for t in texts)


def save(c):
    FIG.mkdir(parents=True, exist_ok=True)
    FIG_ZH.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for ext, meta in [('pdf', {'CreationDate': None, 'ModDate': None}),
                      ('svg', {'Date': None}), ('png', None)]:
        path = FIG / f'{STEM}.{ext}'
        c.fig.savefig(path, dpi=300, facecolor='white', metadata=meta)
        shutil.copyfile(path, FIG_ZH / path.name)
        outputs[path.name] = sha(path)
    return outputs


def fonts(pdf):
    lines = subprocess.run(['pdffonts', str(pdf)], capture_output=True, text=True,
                           check=True).stdout.splitlines()[2:]
    return sorted({line.split()[0].split('+')[-1] for line in lines})


def main():
    configure()
    c = Canvas()
    panel_b(c, counts(), panel_a(c))
    problems, texts = qa(c)
    assert not problems, problems
    outputs = save(c)
    used = fonts(FIG / f'{STEM}.pdf')
    assert set(used) <= {'ArialMT', 'Arial-BoldMT', 'Arial-ItalicMT'}, used
    report = dict(figure=STEM, size_inches=[W, H], problems=problems, fonts=used, texts=texts,
                  outputs=outputs, inputs={str(ROLES.relative_to(ROOT)): sha(ROLES)},
                  script_sha256=sha(__file__), scientific_rerun=False)
    (OUT / 'FIGURE1_QA.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',
                                         encoding='utf-8')
    plt.close(c.fig)
    print(json.dumps({'problems': len(problems), 'fonts': used, 'outputs': sorted(outputs)}))


if __name__ == '__main__':
    main()
