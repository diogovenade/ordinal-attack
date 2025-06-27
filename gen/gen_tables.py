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

def gen_tables(output, rows, super_columns, super_higherisbetter, columns, dfs):
    assert len(super_columns) == len(super_higherisbetter) == len(dfs)
    tables = []
    all_cols = sorted(set().union(*(df[columns].unique() for df in dfs)))
    if 'CE' in all_cols:
        all_cols = ['CE'] + [col for col in all_cols if col != 'CE']
    ncols = len(all_cols)
    for df, super_column, higherisbetter in zip(dfs, super_columns, super_higherisbetter):
        pivoted = df.pivot_table(index=rows, columns=columns, values=super_column)
        pivoted = pivoted.reindex(columns=all_cols)
        pivoted = pivoted.iloc[::-1]
        pivoted.index = pivoted.index.map(lambda x: ' '.join(latex_escape(str(y).replace('_', ' ')) for y in (x if isinstance(x, tuple) else (x,))))
        pivoted.columns = [latex_escape(str(col)) for col in pivoted.columns]
        styled = (
            pivoted.style
            .background_gradient(cmap="RdYlGn" if higherisbetter else "RdYlGn_r", axis=None)
            .format(lambda x: f"{x:.3f}" if pd.notnull(x) else '---')
        )

        pivoted = pivoted.fillna('---').applymap(latex_escape)

        table = styled.to_latex(convert_css=True)
        table = table.split('\n')[1:-2]
        tables.append(table)
    rows_combined = []
    for lines in zip(*tables):
        row = ' & '.join(
            (line[:-3] if i == 0 else '&'.join(line.split('&')[1:])[:-3])
            for i, line in enumerate(lines)
        )
        rows_combined.append(row + r' \\')
    table = '\n'.join(rows_combined)
    with open(output, 'w') as f:
        print(r'\documentclass{standalone}', file=f)
        print(r'\usepackage[table]{xcolor}', file=f)
        print(r'\begin{document}', file=f)
        print(r'\begin{tabular}{|l' + (('|'+'r'*ncols)*len(super_columns)) + '|}', file=f)
        print(r'\hline', file=f)
        print(r'&' + '&'.join(r'\multicolumn{' + str(ncols) + '}{c|}{' + c + '}' for c in super_columns) + r'\\\hline', file=f)
        print(table, file=f)
        print(r'\hline', file=f)
        print(r'\end{tabular}', file=f)
        print(r'\end{document}', file=f)
    os.system('pdflatex ' + output)
    os.remove(output[:-3] + 'aux')
    os.remove(output[:-3] + 'log')

df = pd.read_csv('results.csv')
df['Loss'] = df['Loss'].map(rename_losses).fillna(df['Loss'])
dataset = 'CARSDB' # change dataset as needed

base_df = df[
    (df['Dataset'] == dataset) &
    (df['AttackLoss'].isin(['ModelLoss', 'MeanSquaredError', 'none']))
]

mae_worst = (
    base_df
    .sort_values('MAE', ascending=False)
    .groupby(['Attack', 'Target', 'Loss'], as_index=False)
    .first()
)

qwk_worst = (
    base_df
    .sort_values('QWK', ascending=True)
    .groupby(['Attack', 'Target', 'Loss'], as_index=False)
    .first()
)

gen_tables(
    f'table-{dataset}.tex',
    ['Attack', 'Target'],
    ['MAE', 'QWK'],
    [False, True],
    'Loss',
    [mae_worst, qwk_worst]
)
