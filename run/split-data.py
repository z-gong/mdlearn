#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.append('..')
from mdlearn import preprocessing


def main():
    parser = argparse.ArgumentParser(description='Alkane property fitting demo')
    parser.add_argument('-i', '--input', type=str, help='Data')
    parser.add_argument('-o', '--output', default='fp', type=str, help='Output directory')

    opt = parser.parse_args()

    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

    df = pd.read_csv(opt.input, sep='\s+', header=0)
    smiles_array = df.SMILES.values
    selector = preprocessing.Selector(smiles_array)
    sel_mol = preprocessing.Selector(df.SMILES.unique())
    fold = 5
    sel_mol.kfold_partition(1.0, fold)
    for n in range(fold):
        sel_mol.kfold_use(n)
        mol_train = sel_mol.training_set()
        mol_valid = sel_mol.validation_set()

        mol_train_dict = dict([(s, 1) for s in mol_train])
        mol_valid_dict = dict([(s, 1) for s in mol_valid])

        selector.train_index = np.array([mol_train_dict.get(m, 0) for m in smiles_array], dtype=bool)
        selector.valid_index = np.array([mol_valid_dict.get(m, 0) for m in smiles_array], dtype=bool)
        selector.test_index = np.logical_not(np.logical_or(selector.train_index, selector.valid_index))

        selector.save(opt.output + '/part-%i.txt' % (n + 1))


if __name__ == '__main__':
    main()
