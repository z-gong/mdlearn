#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import shutil
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.append('..')
from mdlearn import fitting, visualize, metrics, preprocessing, validation, dataloader


def main():
    parser = argparse.ArgumentParser(description='Alkane property fitting demo')
    parser.add_argument('-i', '--input', type=str, help='Data')
    parser.add_argument('-f', '--fp', type=str, help='Fingerprints')
    parser.add_argument('-o', '--output', default='out', type=str, help='Output directory')
    parser.add_argument('-t', '--target', default='raw_density', type=str, help='Fitting target')
    parser.add_argument('-p', '--part', default='', type=str, help='Partition cache file')
    parser.add_argument('-l', '--layer', default='16,16', type=str, help='Size of hidden layers')
    parser.add_argument('--gpu', default=1, type=int, help='Using gpu')
    parser.add_argument('--epoch', default="500,1000,1000", type=str, help='Number of epochs')
    parser.add_argument('--batch', default=1000, type=int, help='Batch size')
    parser.add_argument('--lr', default="0.01,0.001,0.0001", type=str, help='Initial learning rate')
    parser.add_argument('--l2', default=0.000, type=float, help='L2 Penalty')
    parser.add_argument('--check', default=50, type=int, help='Number of epoch that do convergence check')
    parser.add_argument('--minstop', default=0.2, type=float, help='Minimum fraction of step to stop')
    parser.add_argument('--maxconv', default=2, type=int, help='Times of true convergence that makes a stop')
    parser.add_argument('--optim', default='adam', type=str, help='optimizer')
    parser.add_argument('--continuation', default=False, type=bool, help='continue training')
    parser.add_argument('--pca', default=-1, type=int, help='dimension to discard')
    parser.add_argument('--sobol', default=-1, type=int, help='dimensions to reduce according to sensitivity analysis')

    opt = parser.parse_args()

    if opt.layer != "":
        layers = list(map(int, opt.layer.split(',')))
    else:
        layers = []

    opt_lr = list(map(float, opt.lr.split(',')))
    opt_epochs = list(map(int, opt.epoch.split(',')))

    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    flog = logging.FileHandler(opt.output + '/log.txt', mode='w')
    flog.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='[%(asctime)s] (%(levelname)s) %(message)s', datefmt='%Y-%d-%m %H:%M:%S')
    flog.setFormatter(formatter)
    clog = logging.StreamHandler()
    clog.setFormatter(formatter)
    logger.addHandler(flog)
    logger.addHandler(clog)

    if sys.platform == 'linux':
        logger.info('Use non-interactive Agg backend for matplotlib on linux')
        matplotlib.use('Agg')

    logger.info('Reading data...')
    datax, datay, data_names = dataloader.load(filename=opt.input, target=opt.target, fps=opt.fp.split(','))

    # Store fingerprint identifier files
    for fp in opt.fp.split(','):
        if os.path.exists(fp + '.idx') and Path(fp).parent.absolute() != Path(opt.output).absolute():
            shutil.copy(fp + '.idx', opt.output)

    logger.info('Selecting data...')
    selector = preprocessing.Selector(datax, datay, data_names)
    if opt.part:
        logger.info('Loading partition file %s' % opt.part)
        selector.load(opt.part)
    else:
        logger.warning("Partition file not found. Using auto-partition instead.")
        selector.partition(0.8, 0.1)
        selector.save(opt.output + '/part.txt')
    trainx, trainy, trainname = selector.training_set()
    validx, validy, validname = selector.validation_set()

    logger.info('Training size = %d, Validation size = %d' % (len(trainx), len(validx)))
    logger.info('X input example: (size=%d) %s' % (len(datax[0]), ','.join(map(str, datax[0]))))
    logger.info('Y input example: (size=%d) %s' % (len(datay[0]), ','.join(map(str, datay[0]))))
    logger.info('Normalizing...')
    scaler = preprocessing.Scaler()
    scaler.fit(trainx)
    scaler.save(opt.output + '/scale.txt')
    normed_trainx = scaler.transform(trainx)
    normed_validx = scaler.transform(validx)

    if opt.sobol != -1:
        with open(opt.output + '/sobol_idx.pkl', 'rb') as file:
            sobol_idx = pickle.load(file)
        normed_trainx, normed_validx = sobol_reduce(normed_trainx, normed_validx, len(normed_trainx[0]) - 2 - opt.sobol, sobol_idx)
        logger.info('Sobol reduced dimension: %d' % (opt.sobol))

    if opt.pca != -1:
        normed_trainx, normed_validx, _ = pca_nd(normed_trainx, normed_validx, len(normed_trainx[0]) - opt.pca, logger)
        logger.info('PCA reduced dimension: %d' % (opt.pca))

    logger.info('Final input length: %d' % (len(normed_trainx[0])))
    logger.info('Building network...')
    logger.info('Hidden layers = %r' % layers)
    logger.info('Learning rate = %s' % opt_lr)
    logger.info('Epochs = %s' % opt_epochs)
    logger.info('L2 penalty = %f' % opt.l2)
    logger.info('Batch size = %d' % opt.batch)

    validy_ = validy.copy()  # for further convenience
    trainy_ = trainy.copy()

    normed_trainx = torch.Tensor(normed_trainx)
    trainy = torch.Tensor(trainy)
    normed_validx = torch.Tensor(normed_validx)

    if opt.gpu:  # store everything to GPU all at once
        logger.info('Using GPU acceleration')
        device = torch.device("cuda:0")
        normed_trainx = normed_trainx.to(device)
        trainy = trainy.to(device)
        normed_validx = normed_validx.to(device)

    if opt.optim == 'sgd':
        optimizer = torch.optim.SGD
    elif opt.optim == 'adam':
        optimizer = torch.optim.Adam
    elif opt.optim == 'rms':
        optimizer = torch.optim.RMSprop
    elif opt.optim == 'ada':
        optimizer = torch.optim.Adagrad

    model = fitting.TorchMLPRegressor(len(normed_trainx[0]), len(trainy[0]), layers, batch_size=opt.batch, is_gpu=opt.gpu != 0,
                                      args_opt={'optimizer'   : optimizer,
                                                'lr'          : opt.lr,
                                                'weight_decay': opt.l2
                                                }
                                      )

    model.init_session()
    if opt.continuation:
        cpt = opt.output + '/model.pt'
        logger.info('Continue training from checkpoint %s' % (cpt))
        model.load(cpt)

    logger.info('Optimizer = %s' % (optimizer))

    header = 'Step Loss MeaSquE MeaSigE MeaUnsE MaxRelE Acc2% Acc5% Acc10%'.split()
    logger.info('%-8s %8s %8s %8s %8s %8s %8s %8s %8s' % tuple(header))

    mse_history = []
    converge_times = 0
    mse_min = None
    model_saved = False
    converged = False
    all_epoch = sum(opt_epochs)
    total_epoch = 0

    for k, each_epoch in enumerate(opt_epochs):
        # implement separated learning rate
        model.reset_optimizer({'optimizer': optimizer, 'lr': opt_lr[k], 'weight_decay': opt.l2})
        for i in range(each_epoch):
            total_epoch += 1
            model.fit_epoch(normed_trainx, trainy)
            if total_epoch % opt.check == 0:
                pred_train = model.predict_batch(normed_trainx)
                pred_valid = model.predict_batch(normed_validx)
                mse = metrics.mean_squared_error(validy_, pred_valid)
                mse_history.append(mse)
                err_line = '%-8i %8.2e %8.2e %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f' % (
                    total_epoch,
                    metrics.mean_squared_error(trainy_, pred_train),
                    mse,
                    metrics.mean_signed_error(validy_, pred_valid) * 100,
                    metrics.mean_unsigned_error(validy_, pred_valid) * 100,
                    metrics.max_relative_error(validy_, pred_valid) * 100,
                    metrics.accuracy(validy_, pred_valid, 0.02) * 100,
                    metrics.accuracy(validy_, pred_valid, 0.05) * 100,
                    metrics.accuracy(validy_, pred_valid, 0.10) * 100)

                logger.info(err_line)

                if mse_min is None:
                    mse_min = mse
                elif mse < mse_min:
                    model.save(opt.output + '/model.pt')
                    model_saved = True
                    mse_min = mse

                if total_epoch > all_epoch * opt.minstop:
                    conv, cur_conv = validation.is_converge(np.array(mse_history), nskip=25)
                    if conv:
                        logger.info('Model converge detected at epoch %d' % total_epoch)
                        converge_times += 1

                    if converge_times >= opt.maxconv and cur_conv:
                        logger.info('Model converged at epoch: %d' % total_epoch)
                        converged = True
                        break

    if not converged:
        logger.warning('Model not converged')

    if not model_saved:
        model.save(opt.output + '/model.pt')

    visualizer = visualize.LinearVisualizer(trainy_.reshape(-1), model.predict_batch(normed_trainx).reshape(-1), trainname, 'train')
    visualizer.append(validy_.reshape(-1), model.predict_batch(normed_validx).reshape(-1), validname, 'valid')
    visualizer.dump(opt.output + '/fit.txt')
    visualizer.dump_bad_molecules(opt.output + '/error-0.10.txt', 'valid', threshold=0.1)
    visualizer.dump_bad_molecules(opt.output + '/error-0.20.txt', 'valid', threshold=0.2)
    visualizer.scatter_yy(savefig=opt.output + '/error-train.png', annotate_threshold=0.1, marker='x', lw=0.2, s=5)
    visualizer.hist_error(savefig=opt.output + '/error-hist.png', label='valid', histtype='step', bins=50)
    plt.show()


def pca_nd(X, X_valid, n, logger):
    X_mol = X  # [:, :-2]
    X_unique = np.unique(X_mol, axis=0)
    pca = PCA(n_components=n)
    pca.fit(X_unique)
    X_transform = pca.transform(X_mol)
    # X_transform = np.c_[X_transform, X[:, -2:]]
    X_valid_transform = X_valid  # [:, :-2]
    X_valid_transform = pca.transform(X_valid_transform)
    # X_valid_transform = np.c_[X_valid_transform, X_valid_transform[:, -2:]]
    logger.info("total variance explained:%.3f" % (pca.explained_variance_ratio_.sum()))
    return X_transform, X_valid_transform, pca.explained_variance_ratio_.sum()


def sobol_reduce(X, X_valid, n, sobol_idx):
    '''  n is the total dimensions left  '''
    X_, X_valid_ = X[:, sobol_idx[-n:]], X_valid[:, sobol_idx[-n:]]
    X = np.c_[X_, X[:, -2:]]
    X_valid = np.c_[X_valid_, X_valid[:, -2:]]
    return X, X_valid


if __name__ == '__main__':
    main()
