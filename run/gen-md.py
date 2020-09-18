#!/bin/env python3

import os
import argparse
import logging
import simtk.openmm as omm
from simtk.openmm import app
from mstools.topology import *
from mstools.forcefield import *
from mstools.simsys import *
from mstools import logger
import mstools.ommhelper as oh
from mstools.utils import histogram
import base64
import math
import numpy as np
import pandas as pd
import mdtraj
import multiprocessing
from mdlearn.gcn.graph import read_msd_files

parser = argparse.ArgumentParser(description='Generate fingerprints')
parser.add_argument('-i', '--input', type=str, nargs='+', required=True, help='Data')
parser.add_argument('-o', '--output', default='md', help='Output directory')
parser.add_argument('-t', '--temp', type=int, default=300, help='Temperature')
parser.add_argument('--nstep', type=int, default=int(5E6), help='MD Steps')
parser.add_argument('--nproc', type=int, default=40, help='Simulations to run in parallel')
parser.add_argument('--analyze', action='store_true', help='Perform distribution analysis instead of running MD')

opt = parser.parse_args()

if not os.path.exists(opt.output):
    os.mkdir(opt.output)

logger.setLevel(logging.WARNING)

ff = ForceField.open('../data/dump-MGI.ppf')


def run_md(mol, T, n_step, name):
    top = Topology([mol])
    top.assign_charge_from_ff(ff)
    system = System(top, ff, transfer_bonded_terms=True, suppress_pbc_warning=True)
    ommsys = system.to_omm_system()
    ommtop = top.to_omm_topology()

    integrator = omm.LangevinIntegrator(T, 2, 0.002)
    platform = omm.Platform.getPlatformByName('Reference')
    sim = app.Simulation(ommtop, ommsys, integrator, platform)
    sim.context.setPositions(top.positions)
    sim.reporters.append(oh.StateDataReporter(f'{opt.output}/{name}-{str(T)}.log', 1000))
    sim.reporters.append(oh.GroReporter(f'{opt.output}/{name}-{str(T)}.gro', 100))
    sim.minimizeEnergy()
    sim.step(n_step)


def analyze(mol, T, name):
    id_pairs = []
    for pairs in mol.get_12_13_14_pairs():
        for p in pairs:
            id_pairs.append([p[0].id, p[1].id])

    edges = np.linspace(0, 0.5, 26)  # 0.5 nm maximum with bin size of 0.02 nm
    index = (edges[1:] + edges[:-1]) / 2
    df = pd.DataFrame(index=index)

    trj = mdtraj.load(f'{opt.output}/{name}-{T}.gro')
    distances = mdtraj.compute_distances(trj, id_pairs)
    distances = list(zip(*distances))
    for (i, j), distance in zip(id_pairs, distances):
        x, y = histogram(distance, bins=edges, normed=True)
        df['%s-%s' % (mol.atoms[i].name, mol.atoms[j].name)] = y
    df.to_csv(f'{opt.output}/{name}-{T}.csv', sep='\t', float_format='%.2f')

    return df


if __name__ == '__main__':
    smiles_list = []
    for inp in opt.input:
        df = pd.read_csv(inp, sep='\s+', header=0)
        for smiles in df.SMILES:
            if smiles not in smiles_list:
                smiles_list.append(smiles)
    name_list = [base64.b64encode(smiles.encode()).decode() for smiles in smiles_list]
    mol_list = read_msd_files([f'{name}.msd' for name in name_list], '../data/msdfiles.zip')

    n_group = math.ceil(len(smiles_list) / opt.nproc)
    n_group = 1
    for i in range(n_group):
        print(f'Run tasks for group {i}')
        jobs = []
        for smiles, name, mol in list(zip(smiles_list, name_list, mol_list))[i * opt.nproc: (i + 1) * opt.nproc]:
            print(f'Run task for {smiles}', [(atom.name, atom.type) for atom in mol.atoms])
            if not opt.analyze:
                p = multiprocessing.Process(target=run_md, args=(mol, opt.temp, opt.nstep, name))
            else:
                p = multiprocessing.Process(target=analyze, args=(mol, opt.temp, name))
            jobs.append(p)
            p.start()
        for job in jobs:
            job.join()
