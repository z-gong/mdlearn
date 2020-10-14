#!/usr/bin/env python3

import os


def main():
    input = '../data/All-npt_rand.txt'
    target = 'einter'
    fp = './out-all-npt/fp_simple'
    out_base = './out-all-npt'
    args_extra = '--head 3,2,1 --pickle ./out-all-npt/ffgcn/graph.pickle'

    for p in range(1, 6):
        for i in range(1, 6):
            part = out_base + '/part-%i.txt' % p
            out = out_base + '/ffgcn-%s_%s' % (p, i)
            cmd = f'python train-ffgcn.py -i {input} -t {target} -f {fp} -p {part} -o {out} {args_extra}'
            print(cmd)
            os.system(cmd)


if __name__ == '__main__':
    main()
