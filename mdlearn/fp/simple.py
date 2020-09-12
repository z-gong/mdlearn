import sys
import pybel
from rdkit.Chem import AllChem as Chem, Descriptors
from rdkit.Chem.rdchem import Mol
import numpy as np

from . import Fingerprint


class SimpleIndexer(Fingerprint):
    name = 'simple'

    def __init__(self):
        super().__init__()

    def get_shortest_wiener(self, rdk_mol: Mol):
        max_shortest = 0
        wiener = 0
        mol = Chem.RemoveHs(rdk_mol)
        n_atoms = mol.GetNumAtoms()
        for i in range(0, n_atoms):
            for j in range(i + 1, n_atoms):
                shortest = len(Chem.GetShortestPath(mol, i, j)) - 1
                wiener += shortest
                max_shortest = max(max_shortest, shortest)
        return max_shortest, int(np.log(wiener) * 10)

    def get_ring_info(self, py_mol):
        # bridged atoms
        bridg_Matcher = pybel.Smarts('[x3]')
        # spiro atoms
        spiro_Matcher = pybel.Smarts('[x4]')
        # linked rings
        RR_Matcher = pybel.Smarts('[R]!@[R]')
        # separated rings
        R_R_Matcher = pybel.Smarts('[R]!@*!@[R]')

        r34 = 0
        r5 = 0
        r6 = 0
        r78 = 0
        rlt8 = 0
        aro = 0
        for r in py_mol.sssr:
            rsize = r.Size()
            if rsize == 3 or rsize == 4:
                r34 += 1
            elif r.IsAromatic():
                aro += 1
            elif rsize == 5:
                r5 += 1
            elif rsize == 6:
                r6 += 1
            elif rsize == 7 or rsize == 8:
                r78 += 1
            else:
                rlt8 += 1

        return len(bridg_Matcher.findall(py_mol)), \
               len(spiro_Matcher.findall(py_mol)), \
               len(RR_Matcher.findall(py_mol)), \
               len(R_R_Matcher.findall(py_mol)), \
               r34, r5, r6, r78, rlt8, aro

    def index(self, smiles):
        rdk_mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        index = [rdk_mol.GetNumAtoms(),
                 int(round(Descriptors.MolWt(rdk_mol), 1) * 10),
                 self.get_shortest_wiener(rdk_mol)[0],
                 Chem.CalcNumRotatableBonds(rdk_mol)]

        # py_mol = pybel.readstring('smi', smiles)
        # index += list(self.get_ring_info(py_mol))

        return np.array(index)

    def index_list(self, smiles_list):
        if self._silent:
            return [self.index(s) for s in smiles_list]

        l = []
        print('Calculate ...')
        for i, s in enumerate(smiles_list):
            if i % 100 == 0:
                sys.stdout.write('\r\t%i' % i)
            l.append(self.index(s))
        print('')

        return l


Fingerprint.register(SimpleIndexer)
