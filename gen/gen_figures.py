import pandas as pd
import os
import re

rename_losses = {
    'CrossEntropy': 'CE',
    'BinomialUnimodal_CE': 'BU',
    'UnimodalNet': 'UN',
    'OrdinalEncoding': 'OE',
}

def latex_escape(s):
    if isinstance(s, str):
        return re.sub(r'([_&%$#{}~^\\])', r'\\\1', s)
    return s

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
    fprint(r'  xlabel={' + latex_escape(xaxis) + '},')
    fprint(r'  ylabel={' + latex_escape(yaxis) + '},')
    fprint(r'  width=12cm, height=8cm,')
    fprint(r'  grid=both,')
    fprint(r'  legend style={at={(1.05,1)}, anchor=north west}')
    fprint(r']')
    for each_color in df[color].unique():
        fprint(r'\addplot+ table {')
        fprint('\n'.join(f'{row[xaxis]} {row[yaxis]}' for ix, row in df[df[color] == each_color].iterrows()))
        fprint('};')
        fprint(r'\addlegendentry{' + latex_escape(str(each_color)) + '}')
    fprint(r'\end{axis}\end{tikzpicture}')
    fprint(r'\end{document}')
    f.close()
    os.system('pdflatex ' + output)
    os.remove(output[:-3] + 'aux')
    os.remove(output[:-3] + 'log')

df = pd.read_csv('gsa.csv') # change to CSV of desired attack
df['Loss'] = df['Loss'].map(rename_losses).fillna(df['Loss'])

for attack_loss in ['CrossEntropy', 'ModelLoss', 'MeanSquaredError']:
    dataset_ix = df['Dataset'] == 'CARSDB' # change to desired dataset
    attack_ix = df['AttackLoss'] == attack_loss
    epsilon_below_015_ix = df['Epsilon'] <= 0.3
    epsilon0_ix = df['Epsilon'] == 0
    if attack_loss != 'MeanSquaredError':
        ix = dataset_ix & epsilon_below_015_ix & ((attack_ix & (df['Target'] == 'none')) | epsilon0_ix)
        gen_pgfplots(f'plot-{attack_loss}-no-target.tex', 'Epsilon', 'MAE', 'Loss', df[ix])
    ix = dataset_ix & epsilon_below_015_ix & ((attack_ix & (df['Target'] == 'next_class')) | epsilon0_ix)
    gen_pgfplots(f'plot-{attack_loss}-next-target.tex', 'Epsilon', 'MAE', 'Loss', df[ix])
    ix = dataset_ix & epsilon_below_015_ix & ((attack_ix & (df['Target'] == 'furthest_class')) | epsilon0_ix)
    gen_pgfplots(f'plot-{attack_loss}-far-target.tex', 'Epsilon', 'MAE', 'Loss', df[ix])
