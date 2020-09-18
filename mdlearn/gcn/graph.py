import torch
import dgl
import numpy as np
from rdkit.Chem import AllChem as Chem
from zipfile import ZipFile
import os
import tempfile
import shutil
import pandas as pd

try:
    from mstools.topology import Topology, UnitCell, Molecule
    from mstools.forcefield import ForceField
    from mstools.simsys import System
except:
    MSTOOLS_FOUND = False
else:
    MSTOOLS_FOUND = True


def read_msd_files(msd_files, parent_dir):
    if not MSTOOLS_FOUND:
        raise ModuleNotFoundError('mstools is required for parsing MSD file')

    tmp_dir = None
    if parent_dir.endswith('.zip'):
        tmp_dir = tempfile.mkdtemp()
        with ZipFile(parent_dir) as zip:
            zip.extractall(tmp_dir)

    mol_list = []
    mol_dict = {}  # cache molecules read from MSD files
    for file in msd_files:
        if file in mol_dict:
            mol = mol_dict[file]
        else:
            mol = Topology.open(os.path.join(tmp_dir or parent_dir, file)).molecules[0]
            mol_dict[file] = mol
        mol_list.append(mol)

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir)

    return mol_list


def read_dist_files(dist_files, parent_dir):
    tmp_dir = None
    if parent_dir.endswith('.zip'):
        tmp_dir = tempfile.mkdtemp()
        with ZipFile(parent_dir) as zip:
            zip.extractall(tmp_dir)

    dist_list = []
    dist_dict = {}  # cache DataFrame
    for file in dist_files:
        if file in dist_dict:
            df = dist_dict[file]
        else:
            df = pd.read_csv(os.path.join(tmp_dir or parent_dir, file), header=0, sep='\s+')
            dist_dict[file] = df
        dist_list.append(df)

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir)

    return dist_list


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
    mol_list = read_msd_files(msd_files, parent_dir)

    types = set()
    for mol in mol_list:
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


def msd2dgl_ff_pairs(msd_files, msd_dir, ff_file, dist_files, dist_dir):
    mol_list = read_msd_files(msd_files, msd_dir)
    top = Topology(mol_list)
    ff = ForceField.open(ff_file)
    top.assign_charge_from_ff(ff)
    system = System(top, ff, transfer_bonded_terms=True)
    top, ff = system.topology, system.ff

    graph_list = []
    feats_node_list = []
    feats_p12_list = []
    feats_p13_list = []
    feats_p14_list = []

    dist_list = read_dist_files(dist_files, dist_dir)
    for mol, df in zip(top.molecules, dist_list):
        pairs12, pairs13, pairs14 = mol.get_12_13_14_pairs()

        graph_data = {}
        for etype, pairs in [('pair12', pairs12), ('pair13', pairs13), ('pair14', pairs14)]:
            u = [p[0].id_in_mol for p in pairs]
            v = [p[1].id_in_mol for p in pairs]
            graph_data.update({('atom', etype, 'atom'): (u + v, v + u)})  # bidirectional
        graph = dgl.heterograph(graph_data)
        graph_list.append(graph)

        feats_node = np.zeros((mol.n_atom, 4))
        for i, atom in enumerate(mol.atoms):
            atype = ff.atom_types[atom.type]
            vdw = ff.get_vdw_term(atype, atype)
            feats_node[i] = vdw.sigma, vdw.epsilon, atom.charge, 0
        for i, improper in enumerate(mol.impropers):
            term = system.improper_terms[id(improper)]
            feats_node[improper.atom1.id_in_mol][-1] = term.k / 10
        feats_node_list.append(feats_node)

        length = len(dist_list[0])
        feats_p12 = np.zeros((len(pairs12) * 2, length))  # bidirectional
        feats_p13 = np.zeros((len(pairs13) * 2, length))  # bidirectional
        feats_p14 = np.zeros((len(pairs14) * 2, length))  # bidirectional
        for feats, pairs in [(feats_p12, pairs12), (feats_p13, pairs13), (feats_p14, pairs14)]:
            for i, pair in enumerate(pairs):
                key = '%s-%s' % (pair[0].name, pair[1].name)
                feats[i][:] = feats[i + len(pairs)][:] = df[key].values()
        feats_p12_list.append(feats_p12)
        feats_p13_list.append(feats_p13)
        feats_p14_list.append(feats_p14)

    return graph_list, feats_node_list, {'pair12': feats_p12_list,
                                         'pair13': feats_p13_list,
                                         'pair14': feats_p14_list
                                         }


def msd2dgl_ff(msd_files, parent_dir, ff_file):
    mol_list = read_msd_files(msd_files, parent_dir)
    top = Topology(mol_list)
    ff = ForceField.open(ff_file)
    top.assign_charge_from_ff(ff)
    system = System(top, ff, transfer_bonded_terms=True)
    top, ff = system.topology, system.ff

    graph_list = []
    feats_node_list = []
    feats_bond_list = []
    feats_angle_list = []
    feats_dihedral_list = []

    for mol in top.molecules:
        feats_node = np.zeros((mol.n_atom, 4))
        feats_bond = np.zeros((mol.n_bond * 2, 2))  # bidirectional
        feats_angle = np.zeros((mol.n_angle * 2, 2))  # bidirectional
        feats_dihedral = np.zeros((mol.n_dihedral * 2, 3))  # bidirectional
        for i, atom in enumerate(mol.atoms):
            atype = ff.atom_types[atom.type]
            vdw = ff.get_vdw_term(atype, atype)
            feats_node[i] = vdw.sigma, vdw.epsilon, atom.charge, 0

        for i, bond in enumerate(mol.bonds):
            term = system.bond_terms[id(bond)]
            inv_k = 1e5 / term.k if not term.fixed else 0
            feats_bond[i] = feats_bond[i + mol.n_bond] = term.length * 10, inv_k

        for i, angle in enumerate(mol.angles):
            term = system.angle_terms[id(angle)]
            inv_k = 100 / term.k if not term.fixed else 0
            feats_angle[i] = feats_angle[i + mol.n_angle] = term.theta / 100, inv_k

        for i, dihedral in enumerate(mol.dihedrals):
            term = system.dihedral_terms[id(dihedral)]
            k1, k2, k3, k4 = term.get_opls_parameters()
            feats_dihedral[i] = feats_dihedral[i + mol.n_dihedral] = k1 / 10, k2 / 10, k3 / 10

        for i, improper in enumerate(mol.impropers):
            term = system.improper_terms[id(improper)]
            feats_node[improper.atom1.id_in_mol][-1] = term.k / 10

        feats_node_list.append(feats_node)
        feats_bond_list.append(feats_bond)
        feats_angle_list.append(feats_angle)
        feats_dihedral_list.append(feats_dihedral)

        bonds = [(bond.atom1.id_in_mol, bond.atom2.id_in_mol) for bond in mol.bonds]
        edges = list(zip(*bonds))
        u = torch.tensor(edges[0] + edges[1])  # bidirectional
        v = torch.tensor(edges[1] + edges[0])
        graph_data = {('atom', 'bond', 'atom'): (u, v)}

        angles = [(angle.atom1.id_in_mol, angle.atom3.id_in_mol) for angle in mol.angles]
        edges = list(zip(*angles))
        u = torch.tensor(edges[0] + edges[1])  # bidirectional
        v = torch.tensor(edges[1] + edges[0])
        graph_data.update({('atom', 'angle', 'atom'): (u, v)})

        dihedrals = [(dihedral.atom1.id_in_mol, dihedral.atom4.id_in_mol) for dihedral in mol.dihedrals]
        edges = list(zip(*dihedrals))
        u = torch.tensor(edges[0] + edges[1])  # bidirectional
        v = torch.tensor(edges[1] + edges[0])
        graph_data.update({('atom', 'dihedral', 'atom'): (u, v)})

        graph = dgl.heterograph(graph_data)
        graph_list.append(graph)

    return graph_list, feats_node_list, {'bond'    : feats_bond_list,
                                         'angle'   : feats_angle_list,
                                         'dihedral': feats_dihedral_list
                                         }


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

        feats = np.zeros((graph.num_nodes(), 9))
        # four elements, four neighbours, one bool denotes aromaticity
        elements = ['C', 'H', 'O', 'N']
        for atom in rdkm.GetAtoms():
            feats[atom.GetIdx()][elements.index(atom.GetSymbol())] = 1
            n_neigh = len(atom.GetNeighbors())
            feats[atom.GetIdx()][len(elements) + n_neigh - 1] = 1
            if atom.IsInRing():
                feats[atom.GetIdx()][-1] = 1
        feats_list.append(feats)

    return graph_list, feats_list
