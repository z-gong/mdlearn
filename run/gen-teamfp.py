#!/usr/bin/env python3

import os
import sys
import argparse
import base64
import pandas as pd
from rdkit import Chem

sys.path.append('..')
from mdlearn.teamfp.mol_io import Msd, Molecule
from mdlearn.teamfp import TeamFP

parser = argparse.ArgumentParser(description='Generate fingerprints')
parser.add_argument('-i', '--input', type=str, help='Data')
parser.add_argument('-o', '--output', default='fp', help='Output directory')

opt = parser.parse_args()

df = pd.read_csv(opt.input, sep='\s+', header=0)
smiles_list = df.SMILES.unique().tolist()

if not os.path.exists(opt.output):
    os.mkdir(opt.output)

if __name__ == '__main__':
    fps = []
    for smiles in smiles_list:
        mol = Msd('msdfiles/%s.msd' % base64.b64encode(smiles.encode()).decode(), ignore_H=True).molecule

        ### Mark Rings
        rd_mol = Chem.MolFromSmiles(smiles)
        for rd_atom in rd_mol.GetAtoms():
            if not rd_atom.IsInRing():
                continue
            atom = mol.atoms[rd_atom.GetIdx()]
            if rd_atom.IsInRingSize(3):
                atom.type_ring = atom.type + '_r3'
            elif rd_atom.IsInRingSize(4):
                atom.type_ring = atom.type + '_r4'
            elif rd_atom.IsInRingSize(5):
                atom.type_ring = atom.type + '_r5'
            elif rd_atom.IsInRingSize(6):
                atom.type_ring = atom.type + '_r6'
            elif rd_atom.IsInRingSize(7) or rd_atom.IsInRingSize(8):
                atom.type_ring = atom.type + '_r78'
            else:
                atom.type_ring = atom.type + '_r>8'
        ###

        fp = TeamFP()
        fp.calc_molecule(mol, radius_list=[0], dihedral=False)
        fps.append(fp)

    fps_new = TeamFP.merge_team_fps(fps, fp_time_limit=200)

    with open(os.path.join(opt.output, 'fp_team'), 'w') as f:
        for i, fp in enumerate(fps_new):
            f.write('%s %s\n' % (smiles_list[i], ','.join(map(str, fp.bit_list))))
