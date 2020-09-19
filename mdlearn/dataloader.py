import numpy as np
import pandas as pd
import os
import shutil
import tempfile
import zipfile

try:
    from mstools.topology import Topology
except:
    MSTOOLS_FOUND = False
else:
    MSTOOLS_FOUND = True


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
        fp_list.append(fp_dict.get(smiles, []))

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


def read_msd_files(msd_files, parent_dir):
    if not MSTOOLS_FOUND:
        raise ModuleNotFoundError('mstools is required for parsing MSD file')

    tmp_dir = None
    if parent_dir.endswith('.zip'):
        tmp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(parent_dir) as zip:
            zip.extractall(tmp_dir)

    mol_list = []
    mol_dict = {}  # cache molecules read from MSD files
    for file in msd_files:
        if file in mol_dict:
            mol = mol_dict[file]
        else:
            mol = Topology.open(os.path.join(tmp_dir or parent_dir, file)).molecules[0]
            mol_dict[file] = mol
        mol_list.append(mol)

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir)

    return mol_list


def read_dist_files(dist_files, parent_dir):
    tmp_dir = None
    if parent_dir.endswith('.zip'):
        tmp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(parent_dir) as zip:
            zip.extractall(tmp_dir)

    dist_list = []
    dist_dict = {}  # cache DataFrame
    for file in dist_files:
        if file in dist_dict:
            df = dist_dict[file]
        else:
            df = pd.read_csv(os.path.join(tmp_dir or parent_dir, file), header=0, sep='\s+')
            dist_dict[file] = df
        dist_list.append(df)

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir)

    return dist_list
