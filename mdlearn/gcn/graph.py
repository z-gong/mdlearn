import torch
import dgl
import numpy as np
from rdkit.Chem import AllChem as Chem
from ..teamfp.mol_io import Msd


def msd2dgl(msd_files, hidden_size):
    '''
    Convert a list of MSD files to a list of DGLGraph.

    The features for each node (atom) is a one-hot vector of the atom types stored in MSD files.

    Parameters
    ----------
    msd_files
    hidden_size

    Returns
    -------

    '''
    types = set()
    mol_list = []
    for file in msd_files:
        msd = Msd()
        msd.read(file)
        mol = msd.molecule
        mol_list.append(mol)
        for atom in mol.atoms:
            types.add(atom.type)
    types = list(sorted(types))

    graph_list = []
    for mol in mol_list:
        bonds = []
        for bond in mol.bonds:
            id1, id2 = bond.atom1.id, bond.atom2.id
            bonds.append((id1, id2))
            bonds.append((id2, id1))
        edges = list(zip(*bonds))
        u = torch.tensor(edges[0])
        v = torch.tensor(edges[1])

        graph = dgl.graph((u, v))
        graph = dgl.add_self_loop(graph)

        graph.ndata['x'] = torch.zeros(graph.num_nodes(), len(types))
        graph.ndata['y'] = torch.zeros(graph.num_nodes(), hidden_size)

        for atom in mol.atoms:
            graph.ndata['x'][atom.id][types.index(atom.type)] = 1

        graph_list.append(graph)

    return graph_list


def rdk2dgl(rdk_mol, hidden_size):
    '''
    Convert a RDKit molecule to a DGLGraph.

    Parameters
    ----------
    rdk_mol
    hidden_size

    Returns
    -------

    '''
    bonds = []
    for bond in rdk_mol.GetBonds():
        id1, id2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds.append((id1, id2))
        bonds.append((id2, id1))
    edges = list(zip(*bonds))
    u = torch.tensor(edges[0])
    v = torch.tensor(edges[1])

    graph = dgl.graph((u, v))
    graph = dgl.add_self_loop(graph)

    graph.ndata['x'] = torch.zeros(graph.num_nodes(), 6)
    graph.ndata['y'] = torch.zeros(graph.num_nodes(), hidden_size)

    for atom in rdk_mol.GetAtoms():
        idx = 0
        if atom.GetAtomicNum() == 6:
            idx = 1
        if atom.GetIsAromatic():
            idx = 2
        else:
            n_neigh = len(atom.GetNeighbors())
            if n_neigh == 3:
                idx = 3
            if n_neigh == 2:
                idx = 4
        graph.ndata['x'][atom.GetIdx()][idx] = 1
        if atom.IsInRing():
            graph.ndata['x'][-1] = 1

    return graph
