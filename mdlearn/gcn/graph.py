import torch
import dgl
import numpy as np
from rdkit.Chem import AllChem as Chem
from ..dataloader import read_msd_files, read_dist_files

try:
    from mstools.topology import Topology
    from mstools.forcefield import ForceField
    from mstools.simsys import System
except:
    MSTOOLS_FOUND = False
else:
    MSTOOLS_FOUND = True


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


def mol2dgl_ff_pairs(mol_list, ff_file, dist_list, distinguish_pairs=False):
    top = Topology(mol_list)
    ff = ForceField.open(ff_file)
    top.assign_charge_from_ff(ff, transfer_bci_terms=True)
    system = System(top, ff, transfer_bonded_terms=True, suppress_pbc_warning=True)
    top, ff = system.topology, system.ff

    graph_list = []
    feats_node_list = []
    feats_p12_list = []
    feats_p13_list = []
    feats_p14_list = []
    feats_edge_length = len(dist_list[0])

    for i, (mol, df) in enumerate(zip(top.molecules, dist_list)):
        if i % 1000 == 0:
            print('Processing molecule %i / %i' % (i, len(dist_list)))

        pairs12, pairs13, pairs14 = mol.get_12_13_14_pairs()

        edges = list(zip(*[(p[0].id_in_mol, p[1].id_in_mol) for p in pairs12]))
        u12, v12 = edges[0] + edges[1], edges[1] + edges[0]
        edges = list(zip(*[(p[0].id_in_mol, p[1].id_in_mol) for p in pairs13]))
        u13, v13 = tuple(), tuple()
        if len(pairs13) > 0:
            u13, v13 = edges[0] + edges[1], edges[1] + edges[0]
        edges = list(zip(*[(p[0].id_in_mol, p[1].id_in_mol) for p in pairs14]))
        u14, v14 = tuple(), tuple()
        if len(pairs14) > 0:
            u14, v14 = edges[0] + edges[1], edges[1] + edges[0]

        if distinguish_pairs:
            graph = dgl.heterograph({('atom', 'pair12', 'atom'): (u12, v12),
                                     ('atom', 'pair13', 'atom'): (u13, v13),
                                     ('atom', 'pair14', 'atom'): (u14, v14),
                                     })
        else:
            uv_self = tuple(range(mol.n_atom))
            graph = dgl.heterograph({('atom', 'pair', 'atom'): (u12 + u13 + u14 + uv_self, v12 + v13 + v14 + uv_self)})

        graph_list.append(graph)

        feats_node = np.zeros((mol.n_atom, 3))
        for i, atom in enumerate(mol.atoms):
            atype = ff.atom_types[atom.type]
            vdw = ff.get_vdw_term(atype, atype)
            feats_node[i] = vdw.sigma, vdw.epsilon, atom.charge
        feats_node_list.append(feats_node)

        feats_p12 = np.zeros((len(pairs12) * 2, feats_edge_length))  # bidirectional
        feats_p13 = np.zeros((len(pairs13) * 2, feats_edge_length))  # bidirectional
        feats_p14 = np.zeros((len(pairs14) * 2, feats_edge_length))  # bidirectional
        for i, pair in enumerate(pairs12):
            key = '%s-%s' % (pair[0].name, pair[1].name)
            feats_p12[i][:] = feats_p12[i + len(pairs12)][:] = df[key].values / 10
        for i, pair in enumerate(pairs13):
            key = '%s-%s' % (pair[0].name, pair[1].name)
            feats_p13[i][:] = feats_p13[i + len(pairs13)][:] = df[key].values / 10
        for i, pair in enumerate(pairs14):
            key = '%s-%s' % (pair[0].name, pair[1].name)
            feats_p14[i][:] = feats_p14[i + len(pairs14)][:] = df[key].values / 10
        feats_p12_list.append(feats_p12)
        feats_p13_list.append(feats_p13)
        feats_p14_list.append(feats_p14)

    if distinguish_pairs:
        feats_edges = {'pair12': [feats[:, 2:13] for feats in feats_p12_list],
                       'pair13': [feats[:, 7:18] for feats in feats_p13_list],
                       'pair14': [feats[:, 12:23] for feats in feats_p14_list],
                       }
    else:
        feats_self_list = [np.zeros((mol.n_atom, feats_edge_length)) for mol in top.molecules]
        feats_edges = {'pair': [np.concatenate(x)[:, 2:23] for x in
                                zip(feats_p12_list, feats_p13_list, feats_p14_list, feats_self_list)]
                       }
    return graph_list, feats_node_list, feats_edges


def msd2dgl_ff(mol_list, ff_file):
    top = Topology(mol_list)
    ff = ForceField.open(ff_file)
    top.assign_charge_from_ff(ff, transfer_bci_terms=True)
    system = System(top, ff, transfer_bonded_terms=True, suppress_pbc_warning=True)
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
