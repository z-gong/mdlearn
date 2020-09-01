#!/usr/bin/env python3

import os

# NIST Tc for All
# cmd1 = 'python gen-fp.py -i ../data/nist-All-tc.txt -e morgan1,simple -o out-all-tc'
# cmd2 = 'python split-data.py -i ../data/nist-All-tc.txt -o out-all-tc'
# cmd3 = 'python train.py -i ../data/nist-All-tc.txt -t tc -f out-all-tc/fp_morgan1,out-all-tc/fp_simple -o out-all-tc/out'

# Simu density for All
cmd1 = 'python gen-fp.py -i ../data/result-ML-All-npt.txt -e morgan1,simple -o out-all-density'
cmd2 = 'python split-data.py -i ../data/result-ML-All-npt.txt -o out-all-density'
cmd311 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-1.txt -o out-all-density/11'
cmd312 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-1.txt -o out-all-density/12'
cmd313 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-1.txt -o out-all-density/13'
cmd314 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-1.txt -o out-all-density/14'
cmd321 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-2.txt -o out-all-density/21'
cmd322 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-2.txt -o out-all-density/22'
cmd323 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-2.txt -o out-all-density/23'
cmd324 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-2.txt -o out-all-density/24'
cmd331 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-3.txt -o out-all-density/31'
cmd332 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-3.txt -o out-all-density/32'
cmd333 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-3.txt -o out-all-density/33'
cmd334 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-3.txt -o out-all-density/34'
cmd341 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-4.txt -o out-all-density/41'
cmd342 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-4.txt -o out-all-density/42'
cmd343 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-4.txt -o out-all-density/43'
cmd344 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-4.txt -o out-all-density/44'
cmd351 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-5.txt -o out-all-density/51'
cmd352 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-5.txt -o out-all-density/52'
cmd353 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-5.txt -o out-all-density/53'
cmd354 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-5.txt -o out-all-density/54'

if __name__ == '__main__':
    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd311)
    os.system(cmd312)
    os.system(cmd313)
    os.system(cmd314)
    os.system(cmd321)
    os.system(cmd322)
    os.system(cmd323)
    os.system(cmd324)
    os.system(cmd331)
    os.system(cmd332)
    os.system(cmd333)
    os.system(cmd334)
    os.system(cmd341)
    os.system(cmd342)
    os.system(cmd343)
    os.system(cmd344)
    os.system(cmd351)
    os.system(cmd352)
    os.system(cmd353)
    os.system(cmd354)
