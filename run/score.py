#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import sys

import mdlearn.dataloader

sys.path.append('..')
from mdlearn import fitting, preprocessing, metrics, visualize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', default='out', help='Model directory')
    parser.add_argument('-i', '--input', help='Data')
    parser.add_argument('-f', '--fp', help='Fingerprints')
    parser.add_argument('-t', '--target', help='Target property')
    parser.add_argument('-p', '--part', help='Partition file')
    parser.add_argument('--gpu', default=1, type=int, help='Using GPU')
    parser.add_argument('--visual', default=1, type=int, help='Visualzation data')
    parser.add_argument('--visualx', default='', help='Extra visualisze on special x')
    parser.add_argument('--dump', default='', help='Output of fitting results')
    parser.add_argument('--featrm', default='', type=str, help='Remove features')

    opt = parser.parse_args()

    model = fitting.TorchMLPRegressor(None, None, [])
    model.is_gpu = opt.gpu == 1
    model.load(opt.dir + '/model.pt')

    scaler = preprocessing.Scaler()
    scaler.load(opt.dir + '/scale.txt')

    if opt.featrm == 'auto':
        featrm = [14, 15, 17, 18, 19, 20, 21, 22]
    elif opt.featrm == '':
        featrm = []
    else:
        featrm = list(map(int, opt.featrm.split(',')))

    datax, datay, data_names = mdlearn.dataloader.load(filename=opt.input, target=opt.obj, fps=opt.fp.split(','), featrm=featrm)

    selector = preprocessing.Selector(datax, datay, data_names)
    selector.load(opt.part)

    trainx, trainy, trainname = selector.training_set()
    validx, validy, validname = selector.validation_set()
    testx, testy, testname = selector.test_set()

    normed_trainx = scaler.transform(trainx)
    normed_validx = scaler.transform(validx)
    normed_testx = scaler.transform(testx)

    trainy = trainy.flatten()
    validy = validy.flatten()
    testy = testy.flatten()

    trainy_est = model.predict_batch(normed_trainx).flatten()
    validy_est = model.predict_batch(normed_validx).flatten()
    testy_est = model.predict_batch(normed_testx).flatten()

    def evaluate_model(y, y_est):
        mse = metrics.mean_squared_error(y, y_est)
        ae = np.average(metrics.abs_absolute_error(y, y_est))
        ave_y = np.average(y)
        ave_y_est = np.average(y_est)
        bias = (ave_y_est - ave_y)

        eval_results = OrderedDict()
        eval_results['MSE'] = mse
        eval_results['RMSE'] = np.sqrt(mse)
        eval_results['AE'] = ae
        eval_results['Max AAE'] = metrics.max_absolute_error(y, y_est)
        eval_results['Bias'] = bias

        eval_results['RRMSE'] = np.sqrt(mse) / np.abs(ave_y)
        eval_results['MARE'] = ae / np.abs(ave_y)
        eval_results['Max ARE'] = metrics.max_relative_error(y, y_est)
        eval_results['RBias'] = bias / np.abs(ave_y)

        eval_results['Accuracy1%'] = metrics.accuracy(y, y_est, 0.01)
        eval_results['Accuracy2%'] = metrics.accuracy(y, y_est, 0.02)
        eval_results['Accuracy5%'] = metrics.accuracy(y, y_est, 0.05)
        eval_results['Accuracy10%'] = metrics.accuracy(y, y_est, 0.1)

        return eval_results

    results = []

    results.append(evaluate_model(trainy, trainy_est))
    results.append(evaluate_model(validy, validy_est))
    results.append(evaluate_model(testy, testy_est))
    results.append(evaluate_model(np.concatenate((trainy, validy, testy)), np.concatenate((trainy_est, validy_est, testy_est))))

    print('Dataset\t%s' % ('\t'.join(results[0].keys())))

    fmt = lambda x: '%.3g' % x

    for name, result in zip(['Training', 'Validation', 'Test', 'Overall'], results):
        print('%s\t%s' % (name, '\t'.join([fmt(v) for v in result.values()])))

    visualizer = visualize.LinearVisualizer(trainy, trainy_est, trainname, 'training')
    visualizer.append(validy, validy_est, validname, 'validation')
    visualizer.append(testy, testy_est, testname, 'test')
    if opt.dump:
        visualizer.dump(opt.dump)

    if opt.visual:
        visualizer.scatter_yy(annotate_threshold=0.1, marker='x', lw=0.2, s=5, figure_name='Value')
        visualizer.scatter_error(annotate_threshold=0.1, marker='x', lw=0.2, s=5, figure_name='Error')
        visualizer.hist_error(label='test', histtype='step', bins=50, figure_name='Error Distribution')

        if opt.visualx:

            for i in map(int, opt.visualx.split(',')):
                visualizer2 = visualize.LinearVisualizer(trainx[:, i], trainy_est - trainy, trainname, 'training')
                visualizer2.append(validx[:, i], validy_est - validy, validname, 'validation')
                visualizer2.append(testx[:, i], testy_est - testy, testname, 'test')
                visualizer2.scatter_yy(ref=None, annotate_threshold=-1, marker='x', lw=0.2, s=5, figure_name=str(i))

        plt.show()


if __name__ == '__main__':
    main()
