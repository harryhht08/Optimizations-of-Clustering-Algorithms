'''
networks.py: contains network definitions (for siamese net,
triplet siamese net, and spectralnet)
'''
from timeit import default_timer

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Subtract

from . import train2
from . import costs
from .layer import stack_layers
from .util import LearningHandler, make_layer_list, train_gen, get_scale

class SiameseNet:
    def __init__(self, inputs, arch, siam_reg, y_true):
        self.orig_inputs = inputs
        # set up inputs
        self.inputs = {
                'A': inputs['Unlabeled'],
                'B': Input(shape=inputs['Unlabeled'].get_shape().as_list()[1:]),
                'Labeled': inputs['Labeled'],
                }

        self.y_true = y_true

        # generate layers
        self.layers = []
        self.layers += make_layer_list(arch, 'siamese', siam_reg)

        # create the siamese net
        self.outputs = stack_layers(self.inputs, self.layers)

        # add the distance layer
        self.distance = Lambda(costs.euclidean_distance, output_shape=costs.eucl_dist_output_shape)([self.outputs['A'], self.outputs['B']])

        #create the distance model for training
        self.net = Model([self.inputs['A'], self.inputs['B']], self.distance)

        # compile the siamese network
        self.net.compile(loss=costs.get_contrastive_loss(m_neg=1, m_pos=0.05), optimizer='rmsprop')

    def train(self, pairs_train, dist_train, pairs_val, dist_val,
            lr, drop, patience, num_epochs, batch_size):
        # create handler for early stopping and learning rate scheduling
        self.lh = LearningHandler(
                lr=lr,
                drop=drop,
                lr_tensor=self.net.optimizer.lr,
                patience=patience)

        # initialize the training generator
        train_gen_ = train_gen(pairs_train, dist_train, batch_size)

        # format the validation data for keras
        validation_data = ([pairs_val[:, 0], pairs_val[:, 1]], dist_val)

        # compute the steps per epoch
        steps_per_epoch = int(len(pairs_train) / batch_size)

        # train the network
        hist = self.net.fit_generator(train_gen_, epochs=num_epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch, callbacks=[self.lh])

        return hist

    def predict(self, x, batch_sizes):
        # compute the siamese embeddings of the input data
        return train2.predict(self.outputs['A'], x_unlabeled=x, inputs=self.orig_inputs, y_true=self.y_true, batch_sizes=batch_sizes)

class SpectralNet:
    def __init__(self, inputs=None, arch=None, spec_reg=None, y_true=None, y_train_labeled_onehot=None,
            n_clusters=None, affinity=None, scale_nbr=None, n_nbrs=None, batch_sizes=None,
            siamese_net=None, x_train=None, have_labeled=False, isUnsupervised=False):

        self.n_clusters = n_clusters
        self.y_true = y_true
        self.y_train_labeled_onehot = y_train_labeled_onehot
        self.inputs = inputs
        self.batch_sizes = batch_sizes
        # generate layers
        self.layers = make_layer_list(arch[:-1], 'spectral', spec_reg)
        self.layers += [
                  {'type': 'tanh',
                   'size': n_clusters,
                   'l2_reg': spec_reg,
                   'name': 'spectral_{}'.format(len(arch)-1)},
                  {'type': 'Orthonorm', 'name':'orthonorm'}
                  ]

        # create spectralnet
        self.outputs = stack_layers(self.inputs, self.layers)
        self.net = Model(inputs=self.inputs['Unlabeled'], outputs=self.outputs['Unlabeled'])

        # DEFINE LOSS

        # generate affinity matrix W according to params
        if affinity == 'siamese':
            input_affinity = tf.concat([siamese_net.outputs['A'], siamese_net.outputs['Labeled']], axis=0)
            x_affinity = siamese_net.predict(x_train, batch_sizes)
        elif affinity in ['knn', 'full'] and isUnsupervised:
            input_affinity = self.inputs['Unlabeled']
            x_affinity = x_train
        elif affinity in ['knn', 'full']:
            input_affinity = tf.concat([self.inputs['Unlabeled'], self.inputs['Labeled']], axis=0)
            x_affinity = x_train

        # calculate scale for affinity matrix
        scale = get_scale(x_affinity, self.batch_sizes['Unlabeled'], scale_nbr)

        # create affinity matrix
        if affinity == 'full':
            W = costs.full_affinity(input_affinity, scale=scale)
        elif affinity in ['knn', 'siamese']:
            W = costs.knn_affinity(input_affinity, n_nbrs, scale=scale, scale_nbr=scale_nbr)

        # if we have labels, use them
        if have_labeled:
            # get true affinities (from labeled data)
            W_true = tf.cast(tf.equal(costs.squared_distance(y_true), 0),dtype='float32')

            # replace lower right corner of W with W_true
            unlabeled_end = tf.shape(self.inputs['Unlabeled'])[0]
            W_u = W[:unlabeled_end, :]                  # upper half
            W_ll = W[unlabeled_end:, :unlabeled_end]    # lower left
            W_l = tf.concat((W_ll, W_true), axis=1)      # lower half
            W = tf.concat((W_u, W_l), axis=0)

            # create pairwise batch distance matrix self.Dy
            self.Dy = costs.squared_distance(tf.concat([self.outputs['Unlabeled'], self.outputs['Labeled']], axis=0))
        else:
            self.Dy = costs.squared_distance(self.outputs['Unlabeled'])

        # define loss
        self.loss = K.sum(W * self.Dy) / (2 * batch_sizes['Unlabeled'])

        # create the train step update
        self.learning_rate = tf.Variable(0., name='spectral_net_learning_rate')
        self.train_step = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=self.net.trainable_weights)

        # initialize spectralnet variables
        # K.get_session().run(tf.variables_initializer(self.net.trainable_weights))
        K.get_session().run(tf.initialize_all_variables())

    def train(self, x_train_unlabeled=None, x_train_labeled=None, x_val_unlabeled=None,
            lr=None, drop=None, patience=None, num_epochs=None, isUnsupervised=False):
        # create handler for early stopping and learning rate scheduling
        self.lh = LearningHandler(
                lr=lr,
                drop=drop,
                lr_tensor=self.learning_rate,
                patience=patience)

        losses = np.empty((num_epochs,))
        val_losses = np.empty((num_epochs,))

        # begin spectralnet training loop
        self.lh.on_train_begin()
        for i in range(num_epochs):

            # train_step(return_var=None, updates=None, x_unlabeled=None, inputs=None, y_true=None,
            #            batch_sizes=None, x_labeled=None, y_labeled=None,
            #            batches_per_epoch=100):

            # train spectralnet
            losses[i] = train2.train_step(
                    return_var=[self.loss],
                    updates=self.net.updates + [self.train_step],
                    x_unlabeled=x_train_unlabeled,
                    inputs=self.inputs,
                    y_true=self.y_true,
                    batch_sizes=self.batch_sizes,
                    x_labeled=x_train_labeled,
                    y_labeled=self.y_train_labeled_onehot,
                    batches_per_epoch=100, isUnsupervised=isUnsupervised)[0]

            print("i: ", i, "loss: ", losses[i])

            if isUnsupervised:
                val_losses[i] = 0.01
            else:
                # get validation loss
                val_losses[i] = train2.predict_sum(
                        self.loss,
                        x_unlabeled=x_val_unlabeled,
                        inputs=self.inputs,
                        y_true=self.y_true,
                        x_labeled=x_train_unlabeled[0:0],
                        y_labeled=self.y_train_labeled_onehot,
                        batch_sizes=self.batch_sizes, isUnsupervised=isUnsupervised)

            # do early stopping if necessary
            if self.lh.on_epoch_end(i, val_losses[i]):
                print('STOPPING EARLY')
                break

            # print training status
            print("Epoch: {}, loss={:2f}, val_loss={:2f}".format(i, losses[i], val_losses[i]))

        return losses[:i], val_losses[:i]

    def fit(self, X = None, plotit=False, verbose=True, data = ''):
        """
        Fits kmedoids with the option for plotting
        """
        start = default_timer()
        predictions = self.predict(X)
        duration = default_timer() - start

        if plotit:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots(1, 1)
            colors = ["b", "g", "r", "c", "m", "y", "k"]
            if self.k > len(colors):
                raise ValueError("we need more colors")

            for i in range(len(self.centers)):
                X_c = X[self.members == i, :]
                ax.scatter(X_c[:, 0], X_c[:, 1], c=colors[i], alpha=0.5, s=30)
                ax.scatter(
                    X[self.centers[i], 0],
                    X[self.centers[i], 1],
                    c=colors[i],
                    alpha=1.0,
                    s=250,
                    marker="*",
                )


        return n, dfp, mlist, duration


    def predict(self, x):
        # test inputs do not require the 'Labeled' input
        inputs_test = {'Unlabeled': self.inputs['Unlabeled'], 'Orthonorm': self.inputs['Orthonorm']}
        return train2.predict(
                    self.outputs['Unlabeled'],
                    x_unlabeled=x,
                    inputs=inputs_test,
                    y_true=self.y_true,
                    x_labeled=x[0:0],
                    y_labeled=self.y_train_labeled_onehot[0:0],
                    batch_sizes=self.batch_sizes)
