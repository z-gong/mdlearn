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
import matplotlib.pyplot as plt
import base64
import numpy as np
import pandas as pd
import mdtraj
import tempfile
from zipfile import ZipFile
import shutil
import multiprocessing

parser = argparse.ArgumentParser(description='Generate fingerprints')
parser.add_argument('-i', '--input', type=str, nargs='+', required=True, help='Data')
parser.add_argument('-o', '--output', default='md', help='Output directory')
parser.add_argument('-t', '--temp', type=int, default=300, help='Temperature')
parser.add_argument('--nstep', type=int, default=int(5E6), help='MD Steps')
parser.add_argument('--nproc', type=int, default=40, help='Simulations to run in parallel')

opt = parser.parse_args()

if not os.path.exists(opt.output):
    os.mkdir(opt.output)

logger.setLevel(logging.WARNING)

ff = ForceField.open('../data/dump-MGI.ppf')


def read_msd_files(msd_files):
    tmp_dir = tempfile.mkdtemp()
    with ZipFile('../data/msdfiles.zip') as zip:
        zip.extractall(tmp_dir)

    topologies = []
    top_dict = {}
    for file in msd_files:
        if file in top_dict:
            top = top_dict[file]
        else:
            top = Topology.open(os.path.join(tmp_dir, file))
            top_dict[file] = top
        topologies.append(top)

    shutil.rmtree(tmp_dir)

    return topologies


def run_md(top, T, n_step, name):
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


def analyze(top, T, name):
    pairs_di = []
    for i, di in enumerate(top.dihedrals):
        pairs_di.append([di.atom1.id, di.atom4.id, i])

    trj = mdtraj.load(f'{name}-{T}.gro')
    distances = mdtraj.compute_distances(trj, np.array(pairs_di)[:, (0, 1)])
    distances = list(zip(*distances))
    for (_, _, i), distance in zip(pairs_di, distances):
        x, y = histogram(distance, bins=20)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        ax.set(title=str(top.dihedrals[i]))


if __name__ == '__main__':
    smiles_set = set()
    for inp in opt.input:
        df = pd.read_csv(inp, sep='\s+', header=0)
        smiles_set.update(df.SMILES.tolist())
    smiles_list = list(sorted(smiles_set))
    name_list = [base64.b64encode(smiles.encode()).decode() for smiles in smiles_list]
    msd_list = [f'{name}.msd' for name in name_list]
    top_list = read_msd_files(msd_list)
    n_job = len(top_list)
    n_parallel = 4

    for i in range(math.ceil(n_job / n_parallel)):
        if i == 1:
            break
        print(f'Run simulations for group {i}')
        jobs = []
        for smiles, name, top in list(zip(smiles_list, name_list, top_list))[i * n_parallel: (i + 1) * n_parallel]:
            print(f'Run simulation for {smiles}', [(atom.name, atom.type) for atom in top.atoms])
            p = multiprocessing.Process(target=run_md, args=(top, opt.temp, opt.nstep, name))
            jobs.append(p)
            p.start()
        for job in jobs:
            job.join()
