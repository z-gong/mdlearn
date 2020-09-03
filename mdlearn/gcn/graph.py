import torch
import dgl
import numpy as np
from rdkit.Chem import AllChem as Chem
from ..teamfp.mol_io import Msd


def msd2dgl(msd_files):
    '''
    Convert a list of MSD files to a list of DGLGraph and node features.

    The features for each node (atom) is a one-hot vector of the atom types stored in MSD files.

    Parameters
    ----------
    msd_files : list of str

    Returns
    -------
    graph_list : list of DGLGraph
    feats_list : list of np.ndarray of shape (n_atom, n_feat)

    '''
    types = set()
    mol_list = []
    for file in msd_files:
        mol = Msd(file).molecule
        mol_list.append(mol)
        for atom in mol.atoms:
            types.add(atom.type)
    types = list(sorted(types))

    graph_list = []
    feats_list = []
    for mol in mol_list:
        bonds = [(bond.atom1.id, bond.atom2.id) for bond in mol.bonds]
        edges = list(zip(*bonds))
        u = torch.tensor(edges[0] + edges[1])  # bidirectional
        v = torch.tensor(edges[1] + edges[0])

        graph = dgl.graph((u, v))
        graph = dgl.add_self_loop(graph)
        graph_list.append(graph)

        feats = np.zeros((graph.num_nodes(), len(types)))
        for atom in mol.atoms:
            feats[atom.id][types.index(atom.type)] = 1
        feats_list.append(feats)

    return graph_list, feats_list


def msd2hetero(msd_files):
    '''
    Convert a list of MSD files to a list of DGLHeteroGraph and node features.

    The features for each node (atom) is a one-hot vector of the atom types stored in MSD files.

    Parameters
    ----------
    msd_files : list of str

    Returns
    -------
    graph_list : list of DGLGraph
    feats_list : list of np.ndarray of shape (n_atom, n_feat)

    '''
    types = set()
    mol_list = []
    for file in msd_files:
        mol = Msd(file).molecule
        mol_list.append(mol)
        for atom in mol.atoms:
            types.add(atom.type)
    types = list(sorted(types))

    graph_list = []
    feats_list = []
    for mol in mol_list:
        bonds = [(bond.atom1.id, bond.atom2.id) for bond in mol.bonds]
        edges = list(zip(*bonds))
        u = torch.tensor(edges[0] + edges[1])  # bidirectional
        v = torch.tensor(edges[1] + edges[0])
        graph_data = {('atom', 'bond', 'atom'): (u, v)}

        angles = [(angle.atom1.id, angle.atom3.id) for angle in mol.angles]
        edges = list(zip(*angles))
        u = torch.tensor(edges[0] + edges[1])  # bidirectional
        v = torch.tensor(edges[1] + edges[0])
        graph_data.update({('atom', 'angle', 'atom'): (u, v)})

        dihedrals = [(dihedral.atom1.id, dihedral.atom4.id) for dihedral in mol.dihedrals]
        edges = list(zip(*dihedrals))
        u = torch.tensor(edges[0] + edges[1])  # bidirectional
        v = torch.tensor(edges[1] + edges[0])
        graph_data.update({('atom', 'dihedral', 'atom'): (u, v)})

        graph = dgl.heterograph(graph_data)
        graph_list.append(graph)

        feats = np.zeros((graph.num_nodes(), len(types)))
        for atom in mol.atoms:
            feats[atom.id][types.index(atom.type)] = 1
        feats_list.append(feats)

    return graph_list, feats_list


def smi2dgl(smiles_list):
    '''
    Convert a list of SMILES to a list of DGLGraph and node features by using RDKit.

    Parameters
    ----------
    smiles_list : list of str

    Returns
    -------
    graph_list : list of DGLGraph
    feats_list : list of np.ndarray of shape (n_atom, n_feat)

    '''

    graph_list = []
    feats_list = []
    for smiles in smiles_list:
        rdkm = Chem.MolFromSmiles(smiles)
        rdkm = Chem.AddHs(rdkm)

        bonds = []
        for bond in rdkm.GetBonds():
            id1, id2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bonds.append((id1, id2))
            bonds.append((id2, id1))
        edges = list(zip(*bonds))
        u = torch.tensor(edges[0])
        v = torch.tensor(edges[1])

        graph = dgl.graph((u, v))
        graph = dgl.add_self_loop(graph)
        graph_list.append(graph)

        feats = np.zeros((graph.num_nodes(), 6))
        for atom in rdkm.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            idx = 1
            if atom.GetIsAromatic():
                idx = 2
            else:
                n_neigh = len(atom.GetNeighbors())
                if n_neigh == 3:
                    idx = 3
                if n_neigh == 2:
                    idx = 4
            feats[atom.GetIdx()][idx] = 1
            if atom.IsInRing():
                feats[atom.GetIdx()][-1] = 1
        feats_list.append(feats)

    return graph_list, feats_list
