import os
import argparse
import logging
import math
import base64
import numpy as np
import sklearn as sk
import torch
import torch.nn.functional as F
import dgl
import matplotlib.pyplot as plt

from mdlearn.gcn.graph import smi2dgl, msd2dgl, msd2hetero
from mdlearn.gcn.model import GCNModel, GATModel
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
parser.add_argument('--batch', default=1000, type=int, help='Batch size')

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
    fp_array, y_array, name_array = dataloader.load(opt.input, opt.target, opt.fp.split(','))
    smiles_list = [name.split()[0] for name in name_array]
    # only take the n_heavy, shortest and n_rotatable from fp_simple
    # fp_array = fp_array[:, (0, 2, 3,)]
    # only take the n_heavy from fp_simple and T, P
    fp_array = fp_array[:, (0, -1)]

    logger.info('Normalizing extra features...')
    scaler = preprocessing.Scaler()
    scaler.fit(fp_array)
    fp_array = scaler.transform(fp_array)

    logger.info('Generating molecular graphs with %s...' % opt.graph)
    msd_list = ['%s.msd' % base64.b64encode(smiles.encode()).decode() for smiles in smiles_list]
    graph_list, feats_node_list, feats_bond_list, feats_angle_list, feats_dihedral_list = \
        msd2hetero(msd_list, '../data/msdfiles.zip', '../data/dump-MGI.ppf')

    logger.info('Selecting data...')
    selector = preprocessing.Selector(smiles_list)
    if opt.part is not None:
        logger.info('Loading partition file %s' % opt.part)
        selector.load(opt.part)
    else:
        logger.warning('Partition file not provided. Using auto-partition instead')
        selector.partition(0.8, 0.2)

    def get_batched_graph_tensor(index, batch_size=None):
        graphs = [graph_list[i] for i in np.where(index)[0]]
        y = y_array[index]
        feats_node = [feats_node_list[i] for i in np.where(index)[0]]
        feats_b = [feats_bond_list[i] for i in np.where(index)[0]]
        feats_a = [feats_angle_list[i] for i in np.where(index)[0]]
        feats_d = [feats_dihedral_list[i] for i in np.where(index)[0]]
        fp_extra = fp_array[index]
        names = name_array[index]

        n_sample = len(graphs)
        if batch_size is None:
            batch_size = n_sample
        batch_size = min(batch_size, n_sample)
        n_batch = n_sample // batch_size
        if n_batch > 1:
            graphs, y, feats_node, feats_b, feats_a, feats_d, fp_extra, names = \
                sk.utils.shuffle(graphs, y, feats_node, feats_b, feats_a, feats_d, fp_extra, names)

        bg_batch = []
        y_batch = []
        feats_node_batch = []
        feats_b_batch = []
        feats_a_batch = []
        feats_d_batch = []
        feats_extra_batch = []
        names_batch = []
        device = torch.device('cuda:0')
        for i in range(n_batch):
            begin = batch_size * i
            end = batch_size * (i + 1) if i != n_batch else -1
            bg_batch.append(dgl.batch(graphs[begin: end]).to(device))
            y_batch.append(torch.tensor(y[begin:end], dtype=torch.float32, device=device))
            feats_node_batch.append(
                torch.tensor(np.concatenate(feats_node[begin:end]), dtype=torch.float32, device=device))
            feats_b_batch.append(
                torch.tensor(np.concatenate(feats_b[begin:end]), dtype=torch.float32, device=device))
            feats_a_batch.append(
                torch.tensor(np.concatenate(feats_a[begin:end]), dtype=torch.float32, device=device))
            feats_d_batch.append(
                torch.tensor(np.concatenate(feats_d[begin:end]), dtype=torch.float32, device=device))
            feats_extra_batch.append(torch.tensor(fp_extra[begin:end], dtype=torch.float32, device=device))
            names_batch.append(names[begin:end])

        return bg_batch, y_batch, feats_node_batch, feats_b_batch, feats_a_batch, feats_d_batch, feats_extra_batch, names_batch

    bg_batch_train, y_batch_train, feats_node_batch_train, feats_b_batch_train, feats_a_batch_train, feats_d_batch_train, feats_extra_batch_train, names_batch_train = \
        get_batched_graph_tensor(selector.train_index, opt.batch)
    bg_valid, y_valid, feats_node_valid, feats_b_valid, feats_a_valid, feats_d_valid, feats_extra_valid, names_valid = \
        get_batched_graph_tensor(selector.valid_index)

    y_train_array = np.concatenate([y.detach().cpu().numpy() for y in y_batch_train])
    names_train = np.concatenate(names_batch_train)
    y_valid_array = y_array[selector.valid_index]

    logger.info('Training size = %d, Validation size = %d' % (len(y_train_array), len(y_valid_array)))
    logger.info('Batch size = %d' % opt.batch)

    in_feats_node = feats_node_list[0].shape[-1]
    in_feats_extra = fp_array[0].shape[-1]
    # model = GCNModel(in_feats_node, opt.embed, extra_feats=in_feats_extra)
    # model = GATModel(in_feats_node, opt.embed, n_head=opt.head, extra_feats=in_feats_extra)
    from mdlearn.gcn.ffmodel import ForceFieldGATModel
    model = ForceFieldGATModel(in_feats_node, feats_bond_list[0].shape[-1], feats_angle_list[0].shape[-1],
                               feats_dihedral_list[0].shape[-1], in_feats_extra, out_dim=opt.embed, n_head=1)
    model.cuda()
    print(model)
    for name, param in model.named_parameters():
        print(name, param.data.shape)

    header = 'Step Loss MeaSquE MeaSigE MeaUnsE MaxRelE Acc2% Acc5% Acc10%'.split()
    logger.info('%-8s %8s %8s %8s %8s %8s %8s %8s %8s' % tuple(header))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrsteps, gamma=opt.lrgamma)
    for epoch in range(opt.epoch):
        model.train()
        if (epoch + 1) % 100 == 0:
            pred_train = []
        for ib in range(len(bg_batch_train)):
            optimizer.zero_grad()
            pred = model(bg_batch_train[ib], feats_node_batch_train[ib], feats_b_batch_train[ib],
                         feats_a_batch_train[ib], feats_d_batch_train[ib], feats_extra_batch_train[ib])
            loss = F.mse_loss(pred, y_batch_train[ib])
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                pred_train.append(pred.detach().cpu().numpy())
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            pred_train = np.concatenate(pred_train)
            pred_valid = model(bg_valid[0], feats_node_valid[0], feats_b_valid[0], feats_a_valid[0], feats_d_valid[0],
                               feats_extra_valid[0]).detach().cpu().numpy()
            mse_train = metrics.mean_squared_error(y_train_array, pred_train)
            mse_valid = metrics.mean_squared_error(y_valid_array, pred_valid)
            err_line = '%-8i %8.2e %8.2e %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f' % (
                epoch + 1, mse_train, mse_valid,
                metrics.mean_signed_error(y_valid_array, pred_valid) * 100,
                metrics.mean_unsigned_error(y_valid_array, pred_valid) * 100,
                metrics.max_relative_error(y_valid_array, pred_valid) * 100,
                metrics.accuracy(y_valid_array, pred_valid, 0.02) * 100,
                metrics.accuracy(y_valid_array, pred_valid, 0.05) * 100,
                metrics.accuracy(y_valid_array, pred_valid, 0.10) * 100)

            logger.info(err_line)
    torch.save(model, opt.output + '/model.pt')

    visualizer = visualize.LinearVisualizer(y_train_array, pred_train, names_train, 'train')
    visualizer.append(y_valid_array, pred_valid, name_array[selector.valid_index], 'valid')
    visualizer.scatter_yy(savefig=opt.output + '/error-train.png', annotate_threshold=0.1, marker='x', lw=0.2, s=5)
    visualizer.hist_error(savefig=opt.output + '/error-hist.png', label='valid', histtype='step', bins=50)
    visualizer.dump(opt.output + '/fit.txt')
    visualizer.dump_bad_molecules(opt.output + '/error-0.10.txt', 'valid', threshold=0.1)
    plt.show()


if __name__ == '__main__':
    main()
