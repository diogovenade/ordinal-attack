import pandas as pd
import os

rename_losses = {
    'CrossEntropy': 'CE',
    'BinomialUnimodal_CE': 'BU',
    'UnimodalNet': 'UN',
    'OrdinalEncoding': 'OE',
}

def gen_tables(output, rows, super_columns, columns, df):
    tables = []
    for super_column in super_columns:
        pivoted = df.pivot_table(index=rows, columns=columns, values=super_column)
        styled = pivoted.style.background_gradient(cmap="RdYlGn",
            axis=None  # normalize gradient across the entire table
        ).format("{:.3f}")
        ncols = pivoted.shape[1]
        table = styled.to_latex(convert_css=True)
        table = table.split('\n')[1:-2]
        tables.append(table)
    for lines in zip(*tables):
        print(lines[0], lines[1])
    table = '\\\\\n'.join(' & '.join(line[:-3] if i == 0 else '&'.join(line.split('&')[1:])[:-3] for i, line in enumerate(lines)) for lines in zip(*tables))
    with open(output, 'w') as f:
        print(r'\documentclass{standalone}', file=f)
        print(r'\usepackage[table]{xcolor}', file=f)
        print(r'\begin{document}', file=f)
        print(r'\begin{tabular}{|l' + (('|'+'r'*ncols)*2) + '|}', file=f)
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
for dataset in df['Dataset'].unique():
    _df = df[
        (df['Dataset'] == dataset) &
        (df['Epsilon'].isin([0, 0.1])) 
    ]
    gen_tables(f'table-{dataset}.tex', 'Attack', ['Accuracy', 'QWK'], 'Loss', _df)
