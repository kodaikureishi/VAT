
from chainer.datasets import tuple_dataset
from chainer import iterators
from chainer.dataset import convert
from chainer import function

import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import initializers
from chainer import reporter

from chainer import Variable


import chainer.dataset.iterator as iterator_module

from chainer import serializers

from chainer import training

from chainer.training import extensions

import chainer

import random
import copy
import os
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

from sklearn.externals import joblib

import argparse

def reset_seed(seed=0):

    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


class MLP(chainer.Chain):

    def __init__(self, n_in=300, n_mid_units=100, n_layers=2, n_out=19):

        super(MLP, self).__init__()

        self.n_layers = n_layers

        self.k = n_out

        initializer = initializers.GlorotUniform()

        with self.init_scope():

            self.input_layer = L.Linear(n_in, n_mid_units, initialW=initializer)

            for i in range(n_layers-1):

                self.add_link('linear%d' % (i + 1),
                              L.Linear(n_mid_units, n_mid_units, initialW=initializer))

            self.output_layer = L.Linear(n_mid_units, n_out, initialW=initializer)


    def __call__(self, x):

        h = self.fwd(x)

        h = self.output_layer(h)

        return h

    def fwd(self, x):

        h = F.leaky_relu(self.input_layer(x))

        for i in range(self.n_layers-1):

            h = F.leaky_relu(self['linear%d' % (i + 1)](h))

        return h


def vat(forward, distance, x, epsilon=0.5, xi=10, Ip=1):

    xp = chainer.cuda.get_array_module(x)

    y = forward(Variable(x))

    y.unchain_backward()

    d = xp.random.normal(size=x.shape, dtype=xp.float32)

    d = d / xp.sqrt(xp.sum(d ** 2, axis=1)).reshape((x.shape[0], 1))

    # print('1', d[0][0])

    for ip in range(Ip):

        d_var = Variable(d.astype(xp.float32))

        y2 = forward(x + xi * d_var)

        kl_loss = distance(y, y2)

        kl_loss.backward()

        d = d_var.grad

        # print(d[0][0])

        d = d / xp.sqrt(xp.sum(d ** 2, axis=1)).reshape((x.shape[0], 1))

    d_var = Variable(d.astype(xp.float32))

    y2 = forward(x + epsilon * d_var)

    return distance(y, y2)



def distance(y0, y1):

    return kl(F.softmax(y0), F.softmax(y1))

def kl(p, q):

    return F.sum(p * F.log((p + 1e-8)/ (q + 1e-8))) / float(len(p.data))


def loss_unlabeled(forward, x, epsilon=0.5, xi=10, Ip=1):

    L = vat(forward=forward, distance=distance, x=x, epsilon=epsilon, xi=xi, Ip=Ip)

    return L

def save_tsne(model, dst, filename, iterator, palet, convertor=convert.concat_examples, device=None, label_name=None):

    @chainer.training.make_extension()
    def visualize_latent(trainer):

        it = copy.copy(iterator)

        x_list = []

        label_list = []

        for batch in it:

            batch_train, batch_teacher = convertor(batch, device=device)

            with function.no_backprop_mode():

                x = model.fwd(batch_train)

                x_list.append(x.data)

                label_list.append(batch_teacher)

        xp = chainer.cuda.get_array_module(model.input_layer.W)

        x_list = xp.concatenate(x_list, axis=0)

        label_list = xp.concatenate(label_list, axis=0)

        x_list = chainer.cuda.to_cpu(x_list)

        label_list = chainer.cuda.to_cpu(label_list)

        size = len(set(label_list))

        from sklearn.manifold import TSNE

        data_embedded = TSNE(n_components=2).fit_transform(x_list)

        save_fig(dst=dst, filename=filename, epoch=trainer.updater.epoch, x=data_embedded, teacher=label_list,
                 k=size, palet=palet, label_map=label_name)

    return visualize_latent

def save_fig(dst, filename, epoch, x, teacher, k, palet, label_map):

    from pylab import rcParams
    rcParams['figure.figsize'] = 10, 10

    preview_dir = '{}/{}'.format(dst, filename)
    preview_path = preview_dir + '/{:0>8}.png'.format(epoch)

    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    for i in range(k):

        data_colored_index = (teacher == i)

        data_colored = x[data_colored_index]

        ax.scatter(data_colored[:,0], data_colored[:,1], c=palet[i], label='{}'.format(label_map[i]))

    ax.set_title('T-SNE embedding')

    ax.grid(True)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)

    fig.savefig(preview_path)




class VAT_Updater(training.StandardUpdater):

    def __init__(self, iterators, model, optimizer, converter=convert.concat_examples,
                 device=None):

        if isinstance(iterators, iterator_module.Iterator):

            iterator = {'main': iterators}

        self._iterators = iterator

        self.model = model

        self._optimizers = {'main': optimizer}

        self.converter = converter

        self.device = device

        self.iteration = 0

    def update_core(self):

        batch = self._iterators['main'].next()

        batch_train, batch_label = self.converter(batch, device=self.device)

        batch_train_true = batch_train[batch_label!=20]
        batch_label_true = batch_label[batch_label!=20]

        # print(len(batch_train))
        # print(len(batch_train_true))

        # batch_train_false = batch_train[batch_label==20]
        # batch_label_false = batch_label[batch_label==20]

        h = self.model(batch_train_true)

        dis_loss = F.softmax_cross_entropy(h, batch_label_true)

        vat_loss = vat(self.model, distance=distance, x=batch_train)

        accuracy = F.accuracy(h, batch_label_true).data

        reporter.report({'dis_loss': dis_loss, 'vat_loss': vat_loss, 'accuracy': accuracy})

        total_loss = dis_loss + vat_loss

        reporter.report({'total_loss': total_loss})

        for optimizer in self._optimizers.values():

            optimizer.target.cleargrads()

        total_loss.backward()

        self._optimizers['main'].update()


class VAT_Evaluation(training.extensions.Evaluator):

    def __init__(self, iterators, model, converter=convert.concat_examples,
                 device=None, access_name='test'):

        if isinstance(iterators, iterator_module.Iterator):

            iterators = {'main': iterators}

        self._iterators = iterators

        self._targets = {'main': model}

        self._converter = converter

        self.device = device

        self.access_name = access_name + '/'

    def evaluate(self):

        iterator = self._iterators['main']

        model = self._targets['main']

        it = copy.copy(iterator)

        summary = reporter.DictSummary()

        observation = {}

        with reporter.report_scope(observation):

            for batch in it:

                batch_train, batch_teacher = self._converter(batch, self.device)

                with function.no_backprop_mode():

                    h = model(batch_train)

                    dis_loss = F.softmax_cross_entropy(h, batch_teacher)

                    accuracy = F.accuracy(h, batch_teacher).data

                    observation[self.access_name + 'accuracy'] = accuracy

                    observation[self.access_name + 'dis_loss'] = dis_loss.data

        summary.add(observation)

        return summary.compute_mean()




def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', '-g', default=-1, type=int)

    parser.add_argument('--batch_size', '-b', default=500)

    parser.add_argument('--n_layers', '-l', default=2, type=int)

    parser.add_argument('--n_mid_units', '-u', default=100, type=int)

    parser.add_argument('--epoch', '-e', default=100, type=int)

    parser.add_argument('--dataset', '-d')

    parser.add_argument('--out', default='result_')

    args = parser.parse_args()

    with open('../' + args.dataset + '.pkl', 'rb') as f:

        dataset = joblib.load(f)

    print('data downlaoded')

    out = args.out + args.dataset

    train_X, train_y, test_X, test_y = dataset

    batch, dim = train_X.shape

    train_data = tuple_dataset.TupleDataset(train_X, train_y)
    test_data = tuple_dataset.TupleDataset(test_X, test_y)

    train_iter = iterators.SerialIterator(dataset=train_data, batch_size=args.batch_size,
                                          repeat=True, shuffle=True)

    test_iter = iterators.SerialIterator(dataset=test_data, batch_size=args.batch_size,
                                         repeat=False, shuffle=False)

    test_iter2 = iterators.SerialIterator(dataset=test_data, batch_size=args.batch_size,
                                         repeat=False, shuffle=False)

    color = ['black', 'silver', 'red', 'sandybrown', 'bisque', 'tan', 'gold', 'lightgoldenrodyellow', 'olivedrab',
             'chartreuse', 'lightseagreen',
             'paleturquoise', 'deepskyblue', 'navy', 'blue', 'mediumpurple', 'plum', 'mediumvioletred', 'aliceblue']

    label_map = ['automotive', 'food', 'entertainment', 'fashion' , 'sports', 'hobbies', 'finance','health & fitness',
                 'home', 'technology', 'travel', 'public service', 'business', 'department', 'no-profit', 'industry',
                 'government', 'pets', 'energy']

    color_map = [colors.cnames[name] for name in color]

    net = MLP(n_in=dim, n_mid_units=args.n_mid_units, n_layers=args.n_layers)

    if args.gpu >= 0:

        chainer.cuda.get_device_from_id(device_id=args.gpu).use()
        net.to_gpu()

    optimizer = optimizers.Adam(alpha=0.001).setup(net)

    updater = VAT_Updater(train_iter, net, optimizer, converter=convert.concat_examples,
                 device=args.gpu)

    def lr_shift():
        if updater.epoch == 5 or updater.epoch == 7:
            optimizer.alpha *= 0.1
        return optimizer.alpha

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out)

    # trainer.extend(extensions.observe_value('lr',lambda _ : lr_shift()), trigger=(1, 'epoch'))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'), trigger=(20, 'epoch'))
    trainer.extend(VAT_Evaluation(test_iter, net, converter=convert.concat_examples,
                 device=args.gpu, access_name='test'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'dis_loss', 'vat_loss', 'total_loss', 'accuracy', 'test/dis_loss','test/accuracy']
    ))
    trainer.extend(extensions.PlotReport(['dis_loss', 'vat_loss', 'total_loss','test/dis_loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['accuracy', 'test/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.ProgressBar())

    trainer.extend(save_tsne(model=net, dst=out, filename='t-sne', iterator=test_iter2, palet=color_map,
                             convertor=convert.concat_examples, device=args.gpu, label_name=label_map), trigger=(1, 'epoch'))

    trainer.run()

    #推論
    # iter_net = MLP(n_in=batch, n_mid_units=args.n_mid_units, n_layers=args.n_layers)
    # serializers.load_npz('/snapshot_epoch-',
    #                      iter_net, path='updater/model:mian/predicator/')



if __name__ == '__main__':
    main()






















