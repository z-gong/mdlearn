#!/usr/bin/env python3

import os

# NIST Tc for All
# cmd1 = 'python gen-fp.py -i ../data/nist-All-tc.txt -e morgan1,simple -o out-all-tc'
# cmd2 = 'python split-data.py -i ../data/nist-All-tc.txt -o out-all-tc'
# cmd3 = 'python train.py -i ../data/nist-All-tc.txt -t tc -f out-all-tc/fp_morgan1,out-all-tc/fp_simple -o out-all-tc/out'

# Simu einter for All
cmd1 = 'python gen-fp.py -i ../data/result-ML-All-npt.txt -e morgan1,simple -o out-all-density'
cmd2 = 'python split-data.py -i ../data/result-ML-All-npt.txt -o out-all-density'
cmd3 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-1.txt -o out-all-density/11'

def main():
    input = '../data/All-npt_rand.txt'
    target = 'einter'
    fp = './out-all-npt/fp_morgan1,./out-all-npt/fp_simple'
    out_base = './out-all-npt'
    args_extra = '-l 32,24,16 --gpu 0'

    for p in range(1, 6):
        for i in range(1, 6):
            part = out_base + '/part-%i.txt' % p
            out = out_base + '/out-%s_%s' % (p, i)
            cmd = f'python train.py -i {input} -t {target} -f {fp} -p {part} -o {out} {args_extra}'
            print(cmd)
            os.system(cmd)


if __name__ == '__main__':
    main()
