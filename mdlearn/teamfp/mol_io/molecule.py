import itertools
from .eqt import eqt_dih_side, eqt_dih_center


class Atom():
    def __init__(self):
        self.id = -1
        self.name = None
        self.type = None
        self.type_ring = None
        self.element = None
        self.charge = None

        self.bonds: [Bond] = []
        self.neighbours: [Atom] = []

    @property
    def is_H(self):
        return self.element == 'H'

    @property
    def n_bond(self):
        return len(self.bonds)

    @property
    def n_neigh_H(self):
        return len([a for a in self.neighbours if a.is_H])

    @property
    def n_neigh_heavy(self):
        return len([a for a in self.neighbours if not a.is_H])

    @property
    def type_dih_side(self):
        return eqt_dih_side.get(self.type, self.type)

    @property
    def type_dih_center(self):
        return eqt_dih_center.get(self.type, self.type)

    def __repr__(self):
        return '<Atom: %s %s>' % (self.name, self.type)


class Bond():
    def __init__(self, atom1: Atom, atom2: Atom):
        self.atom1 = atom1
        self.atom2 = atom2
        atom1.bonds.append(self)
        atom2.bonds.append(self)
        atom1.neighbours.append(atom2)
        atom2.neighbours.append(atom1)

    def __repr__(self):
        return '<Bond: %s %s>' % (self.atom1.name, self.atom2.name)


class Angle():
    def __init__(self, atom1: Atom, atom2: Atom, atom3: Atom):
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3

    def __repr__(self):
        return '<Angle: %s %s %s>' % (self.atom1.name, self.atom2.name, self.atom3.name)


class Dihedral():
    def __init__(self, atom1: Atom, atom2: Atom, atom3: Atom, atom4: Atom):
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3
        self.atom4 = atom4

    def __repr__(self):
        return '<Dihedral: %s %s %s %s>' % (self.atom1.name, self.atom2.name, self.atom3.name, self.atom4.name)

    @property
    def type(self):
        return '%s,%s,%s,%s' % (self.atom1.type, self.atom2.type, self.atom3.type, self.atom4.type)

    @property
    def type_eqt(self):
        return '%s,%s,%s,%s' % (
            self.atom1.type_dih_side, self.atom2.type_dih_center, self.atom3.type_dih_center, self.atom4.type_dih_side)

    def __eq__(self, other):
        return self.atom1 == other.atom4 and self.atom2 == other.atom3


class Molecule():
    def __init__(self, atoms: [Atom]):
        self.atoms: [Atom] = atoms
        self.bonds: [Bond] = []
        self.angles: [Angle] = []
        self.dihedrals: [Dihedral] = []

        for atom in atoms:
            for bond in atom.bonds:
                if bond not in self.bonds:
                    self.bonds.append(bond)
            for (a1, a2) in itertools.combinations(atom.neighbours, 2):
                self.angles.append(Angle(a1, atom, a2))

        for bond in self.bonds:
            atom1, atom2 = bond.atom1, bond.atom2
            for neigh1 in atom1.neighbours:
                for neigh2 in atom2.neighbours:
                    if neigh1 == atom2 or neigh2 == atom1 or neigh1 == neigh2:
                        continue
                    self.dihedrals.append(Dihedral(neigh1, atom1, atom2, neigh2))

    def canonicalize(self):
        pass
