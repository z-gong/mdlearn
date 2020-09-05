import numpy as np
import pandas as pd


def load(filename, target, fps: []):
    """ Load data from file with different temperature and pressure;
        target: 'density'/'einter'/'compress'/'expansion'/'cp'/'hvap'/'st'/'tc'/'dc'
    """

    df = pd.read_csv(filename, sep='\s+', index_col=False)

    other_lists = []
    if 'T' in df.columns:
        other_lists.append(df['T'].values)
    if 'P' in df.columns:
        other_lists.append(df['P'].values)

    fp_dict = {}
    for fp_file in fps:
        d = pd.read_csv(fp_file, sep='\s+', header=None, names=['SMILES', 'fp'], dtype=str)
        for i, row in d.iterrows():
            if row.SMILES not in fp_dict:
                fp_dict[row.SMILES] = list(map(float, row.fp.split(',')))
            else:
                fp_dict[row.SMILES] += list(map(float, row.fp.split(',')))

    fp_list = []
    for smiles in df['SMILES']:
        fp_list.append(fp_dict[smiles])

    ret = np.vstack(fp_list)

    if other_lists == []:
        datax = ret
    else:
        datax = np.hstack([ret] + [s[:, np.newaxis] for s in other_lists])

    names = []
    if 'T' in df.columns and 'P' in df.columns:
        for name, t, p in zip(df['SMILES'], df['T'], df['P']):
            names.append('%s\t%.2e\t%.2e' % (name, t, p))
    elif 'T' in df.columns:
        for name, t in zip(df['SMILES'], df['T']):
            names.append('%s\t%.2e' % (name, t))
    else:
        for name in df['SMILES']:
            names.append(name)

    return datax, df[target].values[:, np.newaxis], np.array(names)
