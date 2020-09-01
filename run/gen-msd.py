#!/usr/bin/env python3

import os
import sys
import argparse
import base64
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Generate fingerprints')
    parser.add_argument('-i', '--input', type=str, help='Data')
    parser.add_argument('-o', '--output', default='msdfiles', help='Output directory')

    opt = parser.parse_args()

    df = pd.read_csv(opt.input, sep='\s+', header=0)
    smiles_list = df.SMILES.unique().tolist()

    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

    sys.path.append('../../ms-tools')
    from mstools.utils import create_mol_from_smiles
    from mstools.wrapper.dff import DFF
    # dff = DFF('/home/gongzheng/apps/DFF/Developing')
    # dff = DFF(r'C:\Users\zheng\Projects\DFF\Developing')
    dff = DFF(r'/Users/zgong/Projects/dff/Developing')
    for smiles in smiles_list:
        filename = base64.b64encode(smiles.encode()).decode()
        mol2 = os.path.join(opt.output, '%s.mol2' % filename)
        msd = os.path.join(opt.output, '%s.msd' % filename)
        py_mol = create_mol_from_smiles(smiles, minimize=False, mol2_out=mol2)
        dff.convert_model_to_msd(mol2, msd)
        dff.set_formal_charge([msd])
        dff.typing([msd])


if __name__ == '__main__':
    main()
