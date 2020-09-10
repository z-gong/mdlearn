import torch
import dgl
import numpy as np
from rdkit.Chem import AllChem as Chem
from zipfile import ZipFile
import os
import tempfile
import shutil

try:
    from mstools.topology import Topology, UnitCell
    from mstools.forcefield import ForceField
    from mstools.simsys import System
except:
    MSTOOLS_FOUND = False
else:
    MSTOOLS_FOUND = True


def _read_msd_files(msd_files, parent_dir):
    if not MSTOOLS_FOUND:
        raise ModuleNotFoundError('mstools is required for parsing MSD file')

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
            mol = Topology.open(os.path.join(tmp_dir or parent_dir, file)).molecules[0]
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


def msd2dgl_ff(msd_files, parent_dir, ff_file):
    mol_list, types = _read_msd_files(msd_files, parent_dir)
    top = Topology(mol_list, cell=UnitCell([3, 3, 3]))
    ff = ForceField.open(ff_file)
    top.assign_charge_from_ff(ff)
    system = System(top, ff, transfer_bonded_terms=True)
    top, ff = system.topology, system.ff

    graph_list = []
    feats_node_list = []
    feats_edge_list = []

    for mol in top.molecules:
        u = list(range(mol.n_atom))  # self loop
        v = list(range(mol.n_atom))

        bonds = [(bond.atom1.id_in_mol, bond.atom2.id_in_mol) for bond in mol.bonds]
        edges = list(zip(*bonds))
        u += edges[0] + edges[1]  # bidirectional
        v += edges[1] + edges[0]

        angles = [(angle.atom1.id_in_mol, angle.atom3.id_in_mol) for angle in mol.angles]
        edges = list(zip(*angles))
        u += edges[0] + edges[1]  # bidirectional
        v += edges[1] + edges[0]

        # dihedrals = [(dihedral.atom1.id_in_mol, dihedral.atom4.id_in_mol) for dihedral in mol.dihedrals]
        # edges = list(zip(*dihedrals))
        # u += edges[0] + edges[1]  # bidirectional
        # v += edges[1] + edges[0]

        graph = dgl.graph((u, v))
        graph_list.append(graph)

        feats_node = np.zeros((mol.n_atom, 4))
        feats_edge = np.zeros((mol.n_atom + (mol.n_bond + mol.n_angle) * 2, 5))  # bidirectional
        for i, atom in enumerate(mol.atoms):
            atype = ff.atom_types[atom.type]
            vdw = ff.get_vdw_term(atype, atype)
            feats_node[i] = vdw.sigma, vdw.epsilon, atom.charge, 0
            feats_edge[i][0] = 1

        for i, bond in enumerate(mol.bonds):
            term = system.bond_terms[id(bond)]
            idxes = np.array([1, 2])
            feats_edge[mol.n_atom + i][idxes] = 1, term.length * 10
            feats_edge[mol.n_atom + i + mol.n_bond][idxes] = 1, term.length * 10

        for i, angle in enumerate(mol.angles):
            term = system.angle_terms[id(angle)]
            idxes = np.array([3, 4])
            feats_edge[mol.n_atom + 2 * mol.n_bond + i][idxes] = 1, term.theta / 100
            feats_edge[mol.n_atom + 2 * mol.n_bond + i + mol.n_angle][idxes] = 1, term.theta / 100

        # for i, dihedral in enumerate(mol.dihedrals):
        #     term = system.dihedral_terms[id(dihedral)]
        #     k1, k2, k3, k4 = term.get_opls_parameters()
        #     idxes = np.array([5, 6, 7, 8])
        #     _shift = mol.n_atom + 2 * mol.n_bond + 2 * mol.n_angle
        #     feats_edge[_shift + i][idxes] = 1, k1 / 10, k2 / 10, k3 / 10
        #     feats_edge[_shift + i + mol.n_dihedral][idxes] = 1, k1 / 10, k2 / 10, k3 / 10

        for i, improper in enumerate(mol.impropers):
            term = system.improper_terms[id(improper)]
            feats_node[improper.atom1.id_in_mol][-1] = term.k / 10

        feats_node_list.append(feats_node)
        feats_edge_list.append(feats_edge)

    return graph_list, feats_node_list, feats_edge_list


def msd2dgl_ff_hetero(msd_files, parent_dir, ff_file):
    mol_list, types = _read_msd_files(msd_files, parent_dir)
    top = Topology(mol_list, cell=UnitCell([3, 3, 3]))
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
        feats_bond = np.zeros((mol.n_bond * 2, 3))  # bidirectional
        feats_angle = np.zeros((mol.n_angle * 2, 3))  # bidirectional
        feats_dihedral = np.zeros((mol.n_dihedral * 2, 3))  # bidirectional
        for i, atom in enumerate(mol.atoms):
            atype = ff.atom_types[atom.type]
            vdw = ff.get_vdw_term(atype, atype)
            feats_node[i] = vdw.sigma, vdw.epsilon, atom.charge, 0

        for i, bond in enumerate(mol.bonds):
            term = system.bond_terms[id(bond)]
            feats_bond[i] = feats_bond[i + mol.n_bond] = term.length * 10, term.fixed, term.k / 1e5

        for i, angle in enumerate(mol.angles):
            term = system.angle_terms[id(angle)]
            feats_angle[i] = feats_angle[i + mol.n_angle] = term.theta / 100, term.fixed, term.k / 100

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

    return graph_list, feats_node_list, feats_bond_list, feats_angle_list, feats_dihedral_list


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
