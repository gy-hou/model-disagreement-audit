"""Minimal DataFrame -> GitHub markdown table (avoids adding a tabulate dependency)."""
def md(df):
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for row in df.itertuples(index=False):
        lines.append('| ' + ' | '.join('' if v is None else (f'{v}' if not isinstance(v, float) else f'{v:g}') for v in row) + ' |')
    return '\n'.join(lines)
