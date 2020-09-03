from .molecule import Atom, Bond, Molecule


class Msd():
    def __init__(self, msd_file, ignore_H=False):
        self.pbc = []
        self.molecule = None
        self._parse(msd_file, ignore_H)

    def _parse(self, msd_file, ignore_H):
        atoms = []

        with open(msd_file) as f:
            lines = f.read().splitlines()

        n = -1
        for line in lines:
            if line.startswith('#'):
                continue

            n += 1
            if n == 0:
                n_atom = int(line)
                continue

            words = line.split()

            if n > 0 and n <= n_atom:
                atom = Atom()
                atom.id = n - 1
                atom.name = words[1]
                atom.type = words[3]
                atom.type_ring = words[3]
                atom.element = ''.join([c for c in words[1] if not c.isdigit()])
                atoms.append(atom)
                continue

            if n == n_atom + 1:
                n_bond = int(line)
                continue

            if n > n_atom + 1 and n <= n_atom + n_bond + 1:
                atom_i = atoms[int(words[0]) - 1]
                atom_j = atoms[int(words[1]) - 1]
                if ignore_H and (atom_i.element == 'H' or atom_j.element == 'H'):
                    continue

                bond = Bond(atom_i, atom_j)

        if ignore_H:
            atoms = list(filter(lambda x: x.element != 'H', atoms))

        self.molecule = Molecule(atoms)
