import logging
import base64
import pandas as pd
from rdkit.Chem import AllChem as Chem
import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv, GATConv
import matplotlib.pyplot as plt

from mdlearn.gcn.graph import rdk2dgl, msd2dgl
from mdlearn.gcn.model import GCNModel, GATModel
from mdlearn import preprocessing, metrics, visualize

HIDDEN_SIZE = 16


def load_data(data_file, *fp_files):
    smiles_list = []
    y_list = []
    df = pd.read_csv(data_file, sep='\s+')
    for row in df.itertuples():
        smiles_list.append(row.SMILES)
        y_list.append(row.tvap)

    fp_dict = {}
    for file in fp_files:
        d = pd.read_csv(file, sep='\s+', header=None, names=['SMILES', 'fp'], dtype=str)
        for row in d.itertuples():
            if row.SMILES not in fp_dict:
                fp_dict[row.SMILES] = list(map(float, row.fp.split(',')))
            else:
                fp_dict[row.SMILES] += list(map(float, row.fp.split(',')))
    fp_list = [fp_dict[smiles] for smiles in df.SMILES]
    fp_array = np.array(fp_list, dtype=np.float32)
    scaler = preprocessing.Scaler()
    scaler.fit(fp_array)
    fp_array = scaler.transform(fp_array)

    return np.array(smiles_list), np.array(y_list), fp_array


def main():
    smiles_array, y_array, fp_array = load_data('../data/nist-CH-tvap.txt', './out-ch-tvap/fp_simple')
    # only take the n_heavy, shortest and n_rotatable from fp_simple
    # fp_array = fp_array[:, (0, 2, 3,)]
    fp_array = fp_array[:, (0,)]

    selector = preprocessing.Selector(smiles_array)
    # selector.partition(0.8, 0.2)
    selector.load('./out-ch-tvap/part-1.txt')

    msd_list = ['msdfiles/%s.msd' % base64.b64encode(smiles.encode()).decode() for smiles in smiles_array]
    msd_graphs = msd2dgl(msd_list, HIDDEN_SIZE)

    def get_graph_tensor(index):
        graph_list = [msd_graphs[i] for i in np.where(index)[0]]
        graph_batched = dgl.batch(graph_list)
        y = torch.tensor(y_array[index], dtype=torch.float32)
        feats_node = graph_batched.ndata['x']
        feats_extra = torch.tensor(fp_array[index], dtype=torch.float32)

        device = torch.device('cuda:0')
        graph_batched = graph_batched.to(device)
        y = y.to(device)
        feats_node = feats_node.to(device)
        feats_extra = feats_extra.to(device)

        return graph_batched, y, feats_node, feats_extra

    graph_train, y_train, feats_node_train, feats_extra_train = get_graph_tensor(selector.training_index)
    graph_valid, y_valid, feats_node_valid, feats_extra_valid = get_graph_tensor(selector.validation_index)

    y_train_array = y_train.detach().cpu().numpy()
    y_valid_array = y_valid.detach().cpu().numpy()

    in_feats_node = feats_node_train.shape[-1]
    in_feats_extra = feats_extra_train.shape[-1]

    # model = GCNModel(in_feats_node, HIDDEN_SIZE, extra_feats=in_feats_extra)
    model = GATModel(in_feats_node, HIDDEN_SIZE, n_head=3, extra_feats=in_feats_extra)
    model.cuda()
    print(model)
    for name, param in model.named_parameters():
        print(name, param.data.shape)

    logger = logging.getLogger('mdlearn')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='[%(asctime)s] (%(levelname)s) %(message)s', datefmt='%Y-%d-%m %H:%M:%S')
    clog = logging.StreamHandler()
    clog.setFormatter(formatter)
    logger.addHandler(clog)

    header = 'Step Loss MeaSquE MeaSigE MeaUnsE MaxRelE Acc2% Acc5% Acc10%'.split()
    logger.info('%-8s %8s %8s %8s %8s %8s %8s %8s %8s' % tuple(header))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2)
    for epoch in range(2500):
        model.train()
        value = model(graph_train, feats_node_train, feats_extra_train)
        loss = F.mse_loss(value, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            predict = model(graph_valid, feats_node_valid, feats_extra_valid).detach().cpu().numpy()
            mse = metrics.mean_squared_error(y_valid_array, predict)
            err_line = '%-8i %8.2e %8.2e %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f' % (
                epoch + 1, loss.detach().cpu().numpy(), mse,
                metrics.mean_signed_error(y_valid_array, predict) * 100,
                metrics.mean_unsigned_error(y_valid_array, predict) * 100,
                metrics.max_relative_error(y_valid_array, predict) * 100,
                metrics.accuracy(y_valid_array, predict, 0.02) * 100,
                metrics.accuracy(y_valid_array, predict, 0.05) * 100,
                metrics.accuracy(y_valid_array, predict, 0.10) * 100)

            logger.info(err_line)

    predict_train = model(graph_train, feats_node_train, feats_extra_train).detach().cpu().numpy()
    predict_valid = model(graph_valid, feats_node_valid, feats_extra_valid).detach().cpu().numpy()
    visualizer = visualize.LinearVisualizer(y_train_array, predict_train, smiles_array[selector.training_index],
                                            'training')
    visualizer.append(y_valid_array, predict_valid, smiles_array[selector.validation_index], 'validation')

    visualizer.scatter_yy(savefig='error-train.png', annotate_threshold=0.1, marker='x', lw=0.2, s=5)
    visualizer.hist_error(savefig='error-hist.png', label='validation', histtype='step', bins=50)
    visualizer.dump('fit.txt')
    visualizer.dump_bad_molecules('error-0.10.txt', 'validation', threshold=0.1)
    plt.show()


if __name__ == '__main__':
    main()
