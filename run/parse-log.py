#!/usr/bin/env python3

import sys
import re
import io
import pandas as pd

def read_log(file):
    with open(file) as f:
        lines = f.read().splitlines()

    string = ''
    for line in lines:
        if line.find('(INFO) Step') > -1:
            string += '\t'.join(line.split()[3:]) + '\n'
        elif re.search('\(INFO\) [0-9]', line):
            string += '\t'.join(line.split()[3:]) + '\n'

    df = pd.read_csv(io.StringIO(string), sep='\s+', header=0, index_col=None)
    return df.loc[df['MeaSquE'].idxmin()]

if __name__ == '__main__':
    datas = []
    for file in sys.argv[1:]:
        row = read_log(file)
        datas.append(row.values)
    df = pd.DataFrame(datas, columns=list(row.index))
    print(df)
