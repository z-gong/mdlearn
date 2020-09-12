""" Higher-level API for fingerprint
"""

import pathlib
import numpy as np
from mdlearn.fp import Fingerprint


class FPEncoder:
    """ Handles higher level processing of fingerprints
    """

    def __init__(self, encoders, fp_name, save_svg=False):
        """ indexer_class: A class that has function index(object) ==> numpy array
        """

        try:
            self.Indexers = [Fingerprint.get_encoder(name) for name in encoders]
        except:
            raise Exception('Available encoders: %s' % Fingerprint.get_encoder_names())

        self.fp_name = fp_name
        self.save_svg = save_svg

    def load_data(self, smiles_list, *other_lists):
        """ Other lists will be vstacked into final fingerprint
        """

        self.smiles_list = smiles_list[:]

        for ol in other_lists:
            assert ol.shape == smiles_list.shape

        self.other_lists = list(other_lists)

    def encode(self, save_fp=True, silent=False):
        ret_list = []

        for Indexer in self.Indexers:
            idxer = Indexer()
            idxer._silent = silent
            if idxer.use_pre_idx_list is not None:
                with open(self.fp_name + '_' + idxer.use_pre_idx_list + '.idx') as f:
                    idxer.pre_idx_list = f.read().splitlines()

            if self.save_svg:
                idxer.svg_dir = pathlib.Path(self.fp_name + '-svg_' + Indexer.name)

            results_list = idxer.index_list(self.smiles_list)

            if save_fp:
                fp_filename = self.fp_name + '_' + Indexer.name
                with open(fp_filename, 'w') as f_fp:
                    for name, result in zip(self.smiles_list, results_list):
                        f_fp.write('%s %s\n' % (name, ','.join(map(str, result))))

            if not idxer.use_pre_idx_list and len(idxer.idx_list) > 0:
                with open(self.fp_name + '_' + Indexer.name + '.idx', 'w') as f_idx:
                    for idx in idxer.idx_list:
                        f_idx.write(f'{idx}\n')

            ret_list.append(np.array(results_list))

        ret = np.hstack(ret_list)

        if self.other_lists == []:
            return ret
        else:
            return np.hstack([ret] + [s[:, np.newaxis] for s in self.other_lists])
