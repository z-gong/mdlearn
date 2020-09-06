import torch
import dgl
import numpy as np
from rdkit.Chem import AllChem as Chem
from zipfile import ZipFile
import os
import tempfile
import shutil
from ..msd import Msd


def _read_msd_files(msd_files, parent_dir):
    tmp_dir = None
    if parent_dir.endswith('.zip'):
        tmp_dir = tempfile.mkdtemp()
        with ZipFile(parent_dir) as zip:
            zip.extractall(tmp_dir)

    types = set()
    mol_list = []
    mol_dict = {}  # cache molecules read from MSD files
    for file in msd_files:
        if file in mol_dict:
            mol = mol_dict[file]
        else:
            mol = Msd(os.path.join(tmp_dir or parent_dir, file)).molecule
            for atom in mol.atoms:
                types.add(atom.type)
            mol_dict[file] = mol
        mol_list.append(mol)
    types = list(sorted(types))

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir)

    return mol_list, types


def msd2dgl(msd_files, parent_dir):
    '''
    Convert a list of MSD files to a list of DGLGraph and node features.

    The features for each node (atom) is a one-hot vector of the atom types stored in MSD files.

    Parameters
    ----------
    msd_files : list of str
    parent_dir : str

    Returns
    -------
    graph_list : list of DGLGraph
    feats_list : list of np.ndarray of shape (n_atom, n_feat)

    '''
    mol_list, types = _read_msd_files(msd_files, parent_dir)

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


def msd2hetero(msd_files, parent_dir):
    '''
    Convert a list of MSD files to a list of DGLHeteroGraph and node features.

    The features for each node (atom) is a one-hot vector of the atom types stored in MSD files.

    Parameters
    ----------
    msd_files : list of str
    parent_dir : str

    Returns
    -------
    graph_list : list of DGLGraph
    feats_list : list of np.ndarray of shape (n_atom, n_feat)

    '''
    mol_list, types = _read_msd_files(msd_files, parent_dir)

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

    rdkm_dict = {}  # cache RDKit molecules generated from SMILES
    for smiles in smiles_list:
        if smiles not in rdkm_dict:
            rdkm = Chem.MolFromSmiles(smiles)
            rdkm = Chem.AddHs(rdkm)
            rdkm_dict[smiles] = rdkm

    graph_list = []
    feats_list = []
    for smiles in smiles_list:
        rdkm = rdkm_dict[smiles]
        bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in rdkm.GetBonds()]
        edges = list(zip(*bonds))
        u = torch.tensor(edges[0] + edges[1])  # bidirectional
        v = torch.tensor(edges[1] + edges[0])

        graph = dgl.graph((u, v))
        graph = dgl.add_self_loop(graph)
        graph_list.append(graph)

        feats = np.zeros((graph.num_nodes(), 6))
        for atom in rdkm.GetAtoms():
            if atom.GetAtomicNum() == 1:
                idx = 0
            else:
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
