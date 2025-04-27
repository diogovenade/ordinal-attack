import pandas as pd
import os

rename_losses = {
    'CrossEntropy': 'CE',
    'BinomialUnimodal_CE': 'BU',
    'UnimodalNet': 'UN',
    'OrdinalEncoding': 'OE',
}

def gen_pgfplots(output, xaxis, yaxis, color, df):
    f = open(output, 'w')
    fprint = lambda x: print(x, file=f)
    fprint(r'\documentclass{standalone}')
    fprint(r'\usepackage{pgfplots}')
    fprint(r'\pgfplotsset{compat=1.18}')
    fprint(r'\begin{document}')
    fprint(r'\begin{tikzpicture}\begin{axis}[')
    fprint(r'  xmin=0, ymin=0,')
    fprint(r'  xtick=data, xticklabel style={/pgf/number format/fixed},')
    fprint(r'  xlabel={' + xaxis + '},')
    fprint(r'  ylabel={' + yaxis + '},')
    fprint(r'  width=12cm, height=8cm,')
    fprint(r'  grid=both')
    fprint(r']')
    for each_color in df[color].unique():
        fprint(r'\addplot+ table {')
        fprint('\n'.join(f'{row[xaxis]} {row[yaxis]}' for ix, row in df[df[color] == each_color].iterrows()))
        fprint('};')
        fprint(r'\addlegendentry{' + each_color + '}')
    fprint(r'\end{axis}\end{tikzpicture}')
    fprint(r'\end{document}')
    f.close()
    os.system('pdflatex ' + output)
    os.remove(output[:-3] + 'aux')
    os.remove(output[:-3] + 'log')

df = pd.read_csv('results.csv')
df['Loss'] = df['Loss'].map(rename_losses).fillna(df['Loss'])

for attack in ['CrossEntropy', 'ModelLoss']:
    _df = df[
        (df['Dataset'] == 'UTKFACE') &
        (df['AttackLoss'].isin(['none', attack])) &
        (df['Epsilon'] <= 0.15)
    ]
    gen_pgfplots(f'plot-{attack}-no-target.tex', 'Epsilon', 'Accuracy', 'Loss', _df[_df['Target'] == 'none'])
    gen_pgfplots(f'plot-{attack}-next-target.tex', 'Epsilon', 'Accuracy', 'Loss', _df[_df['Target'] != 'furthest_class'])
    gen_pgfplots(f'plot-{attack}-far-target.tex', 'Epsilon', 'Accuracy', 'Loss', _df[_df['Target'] != 'next_class'])
