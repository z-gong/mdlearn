#!/usr/bin/env python3

import sys
import os
import argparse
import logging
import pandas as pd

logging.captureWarnings(True)

sys.path.append('..')
from mdlearn import encoding


def main():
    parser = argparse.ArgumentParser(description='Generate fingerprints')
    parser.add_argument('-i', '--input', type=str, help='Data')
    parser.add_argument('-o', '--output', default='fp', help='Output directory')
    parser.add_argument('-e', '--encoder', help='Fingerprint encoder')
    parser.add_argument('--svg', action='store_true', help='Save SVG for fingerprints')

    opt = parser.parse_args()

    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

    df = pd.read_table(opt.input, sep='\s+', header=0)
    smiles_list = df.SMILES.unique().tolist()

    encoders = opt.encoder.split(',')
    encoder = encoding.FPEncoder(encoders, fp_name=opt.output + '/fp', save_svg=opt.svg)
    encoder.load_data(smiles_list)
    encoder.encode()


if __name__ == '__main__':
    main()
