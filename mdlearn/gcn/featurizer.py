import numpy as np
import torch
import dgl
from rdkit.Chem import AllChem as Chem

def rdk2dgl(rdk_mol):
    bonds = []
    for bond in rdk_mol.GetBonds():
        id1, id2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds.append((id1, id2))
        bonds.append((id2, id1))
    edges = list(zip(*bonds))
    u = torch.tensor(edges[0])
    v = torch.tensor(edges[1])

    graph = dgl.graph((u, v))
    dgl.add_self_loop(graph)

    graph.ndata['x'] = torch.zeros(graph.num_nodes(), 6)
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

def get_shortest_wiener(rdk_mol):
    wiener = 0
    max_shortest = 0
    mol = Chem.RemoveHs(rdk_mol)
    n_atoms = mol.GetNumAtoms()
    for i in range(0, n_atoms):
        for j in range(i + 1, n_atoms):
            shortest = len(Chem.GetShortestPath(mol, i, j)) - 1
            wiener += shortest
            max_shortest = max(max_shortest, shortest)
    return max_shortest, int(np.log(wiener) * 10)

def get_extra_features(rdk_mol):
    shortest = get_shortest_wiener(rdk_mol)[0]
    heavy = rdk_mol.GetNumHeavyAtoms()
    rotatable = Chem.CalcNumRotatableBonds(rdk_mol)

    return shortest, heavy, rotatable

