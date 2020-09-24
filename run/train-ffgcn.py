import os
import argparse
import logging
import base64
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import dgl
import matplotlib.pyplot as plt

from mdlearn.gcn.graph import msd2dgl_ff, mol2dgl_ff_pairs
from mdlearn.gcn.model_ff import ForceFieldGATModel
from mdlearn import preprocessing, metrics, visualize, dataloader

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='Data')
parser.add_argument('-f', '--fp', type=str, help='Fingerprints')
parser.add_argument('-o', '--output', default='out', type=str, help='Output directory')
parser.add_argument('-t', '--target', default='density', type=str, help='Fitting target')
parser.add_argument('-p', '--part', type=str, help='Partition cache file')
parser.add_argument('-g', '--graph', default='msd', type=str, choices=['msd', 'rdk'], help='Graph type')
parser.add_argument('--embed', default=16, type=int, help='Size of graph embedding')
parser.add_argument('--head', default=3, type=int, help='Heads of GAT network')
parser.add_argument('--epoch', default=1600, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate')
parser.add_argument('--lrsteps', default=400, type=int, help='Scale learning rate every these steps')
parser.add_argument('--lrgamma', default=0.2, type=float, help='Scaling factor for learning rate')
parser.add_argument('--l2', default=0.000, type=float, help='L2 Penalty')
parser.add_argument('--check', default=100, type=int, help='Number of epoch that do convergence check')
parser.add_argument('--batch', default=1000, type=int, help='Approximate batch size')

opt = parser.parse_args()

if not os.path.exists(opt.output):
    os.makedirs(opt.output)

logger = logging.getLogger('mdlearn')
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='[%(asctime)s] (%(levelname)s) %(message)s', datefmt='%Y-%d-%m %H:%M:%S')
clog = logging.StreamHandler()
clog.setFormatter(formatter)
flog = logging.FileHandler(opt.output + '/log.txt', mode='w')
flog.setFormatter(formatter)
logger.addHandler(clog)
logger.addHandler(flog)


def main():
    logger.info('Reading data and extra features...')
    fp_files = [] if opt.fp is None else opt.fp.split(',')
    fp_extra, y_array, name_array = dataloader.load(opt.input, opt.target, fp_files)
    smiles_list = [name.split()[0] for name in name_array]

    _pickle_file = opt.output + '/graph.pickle'
    if os.path.exists(_pickle_file):
        logger.info('Unpickling graph and features...')
        with open(_pickle_file, 'rb') as f:
            graph_list, feats_node_list, feats_edges_list = pickle.load(f)
    else:
        logger.info('Reading MSD files...')
        msd_files = ['%s.msd' % base64.b64encode(smiles.encode()).decode() for smiles in smiles_list]
        mol_list = dataloader.read_msd_files(msd_files, '../data/msdfiles.zip')

        logger.info('Reading pair distribution files...')
        dist_files = ['%s-300.csv' % base64.b64encode(smiles.encode()).decode() for smiles in smiles_list]
        dist_list = dataloader.read_dist_files(dist_files, '../data/distfiles.zip')

        logger.info('Generating molecular graphs...')
        graph_list, feats_node_list, feats_edges_list = mol2dgl_ff_pairs(mol_list, '../data/dump-MGI.ppf', dist_list)

        logger.info('Pickling graph and features...')
        with open(_pickle_file, 'wb') as f:
            pickle.dump([graph_list, feats_node_list, feats_edges_list], f)

    logger.info('Node feature example: (size=%i) %s' % (
        len(feats_node_list[0][0]), ','.join(map(str, feats_node_list[0][0]))))
    for edge_type, feats_list in feats_edges_list.items():
        logger.info('Edge %s feature example: (size=%i) %s' % (
            edge_type, len(feats_list[0][0]), ','.join(map(str, feats_list[0][0]))))
    logger.info('Extra graph feature example: (size=%i) %s' % (len(fp_extra[0]), ','.join(map(str, fp_extra[0]))))
    logger.info('Output example: (size=%d) %s' % (len(y_array[0]), ','.join(map(str, y_array[0]))))

    if fp_extra.shape[-1] > 0:
        logger.info('Normalizing extra graph features...')
        scaler = preprocessing.Scaler()
        scaler.fit(fp_extra)
        scaler.save(opt.output + '/scale.txt')
        fp_extra = scaler.transform(fp_extra)

    logger.info('Selecting data...')
    selector = preprocessing.Selector(smiles_list)
    if opt.part is not None:
        logger.info('Loading partition file %s' % opt.part)
        selector.load(opt.part)
    else:
        logger.warning('Partition file not provided. Using auto-partition instead')
        selector.partition(0.8, 0.2)

    device = torch.device('cuda:0')
    # batched data for training set
    data_list = [[data[i] for i in np.where(selector.train_index)[0]] for data in
                 [graph_list, y_array, feats_node_list] + list(feats_edges_list.values()) +
                 [fp_extra, name_array, smiles_list]
                 ]
    n_batch, data_list_batched = preprocessing.separate_batches(data_list[:-1], opt.batch, data_list[-1])
    graphs_batch, y_batch, feats_node_batch = data_list_batched[:3]
    feats_edges_batch = data_list_batched[3:-2]
    feats_extra_batch, names_batch = data_list_batched[-2:]
    bg_batch_train = [dgl.batch(graphs).to(device) for graphs in graphs_batch]
    y_batch_train = [torch.tensor(y, dtype=torch.float32, device=device) for y in y_batch]
    feats_node_batch_train = [torch.tensor(np.concatenate(feats), dtype=torch.float32, device=device)
                              for feats in feats_node_batch]
    feats_edges_batch_train = []
    for feats_edges in zip(*feats_edges_batch):
        _t_edges = (torch.tensor(np.concatenate(feats), dtype=torch.float32, device=device) for feats in feats_edges)
        feats_edges_batch_train.append(dict(zip(feats_edges_list.keys(), _t_edges)))
    feats_extra_batch_train = [torch.tensor(feats, dtype=torch.float32, device=device) for feats in feats_extra_batch]
    # for plot
    y_train_array = np.concatenate(y_batch)
    names_train = np.concatenate(names_batch)

    # data for validation set
    data_list = [[data[i] for i in np.where(selector.valid_index)[0]] for data in
                 [graph_list, y_array, feats_node_list] + list(feats_edges_list.values()) + [fp_extra, name_array]
                 ]
    graphs, y, feats_node = data_list[:3]
    feats_edges = data_list[3:-2]
    feats_extra, names_valid = data_list[-2:]
    bg_valid = dgl.batch(graphs).to(device)
    feats_node_valid = torch.tensor(np.concatenate(feats_node), dtype=torch.float32, device=device)
    _t_edges = (torch.tensor(np.concatenate(feats), dtype=torch.float32, device=device) for feats in feats_edges)
    feats_edges_valid = dict(zip(feats_edges_list.keys(), _t_edges))
    feats_extra_valid = torch.tensor(feats_extra, dtype=torch.float32, device=device)
    # for plot
    y_valid_array = y_array[selector.valid_index]

    logger.info('Training size = %d, Validation size = %d' % (len(y_train_array), len(y_valid_array)))
    logger.info('Batches = %d, Batch size ~= %d' % (n_batch, opt.batch))

    in_feats_node = feats_node_list[0].shape[-1]
    in_feats_edges = {edge_type: feats_edge[0].shape[-1] for edge_type, feats_edge in feats_edges_list.items()}
    in_feats_extra = fp_extra[0].shape[-1]
    model = ForceFieldGATModel(in_feats_node, in_feats_edges, in_feats_extra, out_dim=opt.embed, n_head=opt.head)
    model.cuda()
    print(model)
    for name, param in model.named_parameters():
        print(name, param.data.shape)

    header = 'Step MaxRE(t) Loss MeaSquE MeaSigE MeaUnsE MaxRelE Acc2% Acc5% Acc10%'.split()
    logger.info('%-8s %8s %8s %8s %8s %8s %8s %8s %8s %8s' % tuple(header))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrsteps, gamma=opt.lrgamma)
    for epoch in range(opt.epoch):
        model.train()
        if (epoch + 1) % opt.check == 0:
            pred_train = [None] * n_batch
        for ib in np.random.permutation(n_batch):
            optimizer.zero_grad()
            pred = model(bg_batch_train[ib], feats_node_batch_train[ib], feats_edges_batch_train[ib],
                         feats_extra_batch_train[ib])
            loss = F.mse_loss(pred, y_batch_train[ib])
            loss.backward()
            optimizer.step()
            if (epoch + 1) % opt.check == 0:
                pred_train[ib] = pred.detach().cpu().numpy()
        scheduler.step()

        if (epoch + 1) % opt.check == 0:
            model.eval()
            pred_train = np.concatenate(pred_train)
            pred_valid = model(bg_valid, feats_node_valid, feats_edges_valid, feats_extra_valid).detach().cpu().numpy()
            err_line = '%-8i %8.1f %8.2e %8.2e %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f' % (
                epoch + 1,
                metrics.max_relative_error(y_train_array, pred_train) * 100,
                metrics.mean_squared_error(y_train_array, pred_train),
                metrics.mean_squared_error(y_valid_array, pred_valid),
                metrics.mean_signed_error(y_valid_array, pred_valid) * 100,
                metrics.mean_unsigned_error(y_valid_array, pred_valid) * 100,
                metrics.max_relative_error(y_valid_array, pred_valid) * 100,
                metrics.accuracy(y_valid_array, pred_valid, 0.02) * 100,
                metrics.accuracy(y_valid_array, pred_valid, 0.05) * 100,
                metrics.accuracy(y_valid_array, pred_valid, 0.10) * 100)

            logger.info(err_line)
    torch.save(model, opt.output + '/model.pt')

    visualizer = visualize.LinearVisualizer(y_train_array.reshape(-1), pred_train.reshape(-1), names_train, 'train')
    visualizer.append(y_valid_array.reshape(-1), pred_valid.reshape(-1), names_valid, 'valid')
    visualizer.dump(opt.output + '/fit.txt')
    visualizer.dump_bad_molecules(opt.output + '/error-0.10.txt', 'valid', threshold=0.1)
    visualizer.scatter_yy(savefig=opt.output + '/error-train.png', annotate_threshold=0.1, marker='x', lw=0.2, s=5)
    visualizer.hist_error(savefig=opt.output + '/error-hist.png', label='valid', histtype='step', bins=50)
    plt.show()


if __name__ == '__main__':
    main()
