from mdlearn.fp import Fingerprint
from .mol_io.molecule import Atom, Molecule


class TeamFP(Fingerprint):
    def __init__(self):
        super().__init__()

    def replace_type_center(self, type):
        d = {
        }
        return d.get(type, type)

    def replace_type_neigh(self, type):
        d = {
        }
        return d.get(type, type)

    def calc_atom(self, atom: Atom, radius: int) -> str:
        if radius == 0:
            return '0.' + atom.type_ring

        if radius == 1:
            types_neigh = [self.replace_type_neigh(a.type_ring) for a in atom.neighbours]
            types_neigh.sort()
            return '1.' + ','.join([self.replace_type_center(atom.type_ring)] + types_neigh)

        if radius == 2:
            raise Exception('radius > 1 not supported')

    def calc_molecule(self, molecule: Molecule, radius_list: [int], dihedral=False):
        fp = {}
        for r in radius_list:
            for atom in molecule.atoms:
                idx = self.calc_atom(atom, r)
                if idx not in fp:
                    fp[idx] = 0
                fp[idx] += 1

        if dihedral:
            for dih in molecule.dihedrals:
                idx = 'd.' + dih.type_eqt
                if idx not in fp:
                    fp[idx] = 0
                fp[idx] += 1

        self.bit_count = fp

    @staticmethod
    def merge_team_fps(fps: [Fingerprint], fp_time_limit=1) -> [Fingerprint]:
        idx_set = set()
        fps_new = []
        for fp in fps:
            idx_set.update(fp.bit_count.keys())

        idx_list = list(sorted(idx_set))

        ### Filter idx based on times
        idx_times = dict([(k, 0) for k in idx_list])
        for fp in fps:
            for k, v in fp.bit_count.items():
                idx_times[k] += 1
        print('%i identifiers total' % (len(idx_times)))
        # print(sorted(idx_times.items(), key=lambda x: x[1], reverse=True))
        for k, v in sorted(idx_times.items(), key=lambda x: x[1], reverse=True):
            print(k, v)

        ### Keep all the idx with radius equal to 0
        idx_list = [k for (k, v) in idx_times.items() if k.startswith('0') or v >= fp_time_limit]
        print('%i identifiers remain' % (len(idx_list)))
        print(idx_list)

        for fp in fps:
            bit_count = dict([(k, 0) for k in idx_list])
            for k, v in fp.bit_count.items():
                if k in idx_list:
                    bit_count[k] = v

            fp_new = Fingerprint()
            fp_new.bit_count = bit_count
            fps_new.append(fp_new)

        return fps_new
