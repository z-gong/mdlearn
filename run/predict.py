#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np

sys.path.append('..')
from mdlearn.utils import ml_predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', default='out', help='Model directory')
    parser.add_argument('-i', '--input', default='CC,207,100', help='Molecule,T,P')
    parser.add_argument('-e', '--encoder', help='Fingerprint encoder')
    parser.add_argument('--batch', default='', help='Batch input file')

    opt = parser.parse_args()

    encoders = opt.encoder.split(',')

    smiles_list = []
    t_list = []
    p_list = []

    if not opt.batch:
        words = opt.input.split(',')
        smiles_list.append(words[0])
        if len(words) >= 2:
            t_list.append(float(words[1]))
        if len(words) >= 3:
            p_list.append(float(words[2]))

    else:
        with open(opt.batch, 'r') as f:
            for line in f.readlines():
                words = line.split(',')
                smiles_list.append(words[0])
                if len(words) >= 2:
                    t_list.append(float(words[1]))
                if len(words) >= 3:
                    p_list.append(float(words[2]))

    datay = ml_predict(opt.dir, smiles_list, t_list=t_list, p_list=p_list, encoders=encoders)

    print('SMILES\tT\tP\tResult')
    for s_, t_, p_, y_ in zip(smiles_list, t_list or [0] * len(smiles_list), p_list or [0] * len(smiles_list), datay):
        print('%s\t%g\t%g\t%.3g' % (s_, t_, p_, y_))


if __name__ == '__main__':
    main()
