import csv
import logging
from collections import Counter

import matplotlib.pylab as plt
import pandas as pd

from gam_package.medoids_algorithms.k_medoids import KMedoids
from gam_package.distance_functions.kendall_tau_distance import mergeSortDistance
from gam_package.distance_functions.spearman_distance import spearman_squared_distance
from gam_package.medoids_algorithms.parallel_medoids import ParallelMedoids
from gam_package.plot_functions.plot import parallelPlot, radarPlot, facetedRadarPlot, silhouetteAnalysis
from gam_package.medoids_algorithms.ranked_medoids import RankedMedoids
from gam_package.medoids_algorithms.bandit_pam import BanditPAM

import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from sklearn.cluster import KMeans

from keras.layers import Input

from spectral_clustering.core import networks2
from spectral_clustering.core.util import get_cluster_sols

'''
Expected run times on a GTX 1080 GPU:
MNIST: 1 hr
Reuters: 2.5 hrs
cc: 15 min
'''

import sys, os
# add directories in src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import argparse
from collections import defaultdict

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO
)


class GAM:
    """Generates global attributions

    Args:
        k (int): number of clusters and centroids to form, default=2
        attributions_path (str): path for csv containing local attributions
        cluster_method: None, or callable, default=None
            None - use GAM library routines for k-medoids clustering
            callable - user provided external function to perform clustering
        distance: {â€˜spearmanâ€™, â€˜kendallâ€™}  distance metric used to compare attributions, default='spearman'
        use_normalized (boolean): whether to use normalized attributions in clustering, default='True'
        scoring_method (callable) function to calculate scalar representing goodness of fit for a given k, default=None
        max_iter (int): maximum number of iteration in k-medoids, default=100
        tol (float): tolerance denoting minimal acceptable amount of improvement, controls early stopping, default=1e-3
    """

    def __init__(
        self,
        n_clusters=2,
        attributions_path='data/mushroom-attributions-200-samples.csv',
        cluster_method="parallel medoids",
        distance="euclidean",
        use_normalized=True,
        scoring_method=None,
        max_iter=100,
        tol=1e-3,
    ):
        self.attributions_path = attributions_path # file path for csv dataset
        self.cluster_method = cluster_method # string representing appropriate k-medoids algorithm

        self.distance = distance # string specifying appropriate dissimilarity metric
        if self.distance == "euclidean":
            self.distance_function = self.euclidean_distance
        elif self.distance == "spearman":
            self.distance_function = spearman_squared_distance
        elif self.distance == "kendall":
            self.distance_function = mergeSortDistance
        else:
            self.distance_function = (
                distance
            )  # assume this is metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS

        self.scoring_method = scoring_method

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

        self.attributions = None # later initialized to pandas dataframe holding csv data
        self.use_normalized = use_normalized # (boolean): whether to use normalized attributions in clustering, default='True'
        self.clustering_attributions = None #
        self.feature_labels = None

        self.subpopulations = None
        self.subpopulation_sizes = None
        self.explanations = None
        self.score = None

    def euclidean_distance(self, data1, data2):
        """example distance function"""
        return np.sqrt(np.sum((data1 - data2) ** 2))

    @staticmethod
    def normalize(attributions):
        """
        Normalizes attributions by via absolute value
            normalized = abs(a) / sum(abs(a))

        Args:
            attributions (numpy.ndarray): for example, [(2, 8), (1, 9)]

        Returns: normalized attributions (numpy.ndarray). For example, [(.2, .8), (.1, .9)]
        """
        # keepdims for division broadcasting
        total = np.abs(attributions).sum(axis=1, keepdims=True)

        return np.abs(attributions) / total

    def _read_local(self):
        """
        Reads attribution values and feature labels from csv

        Returns:
            attributions (numpy.ndarray): for example, [(.2, .8), (.1, .9)]
            feature labels (tuple of labels): ("height", "weight")
        """
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
        raw_data = urllib.urlopen(url)
        with open('csv_file.csv', 'wb') as file:
            file.write(raw_data.read())
        """
        # use numpy to process data from csv file, the header labels are discarded
        self.attributions = np.genfromtxt(
            self.attributions_path, dtype=float, delimiter=",", skip_header=1
        )

        # extract the feature labels
        with open(self.attributions_path) as attribution_file:
            self.feature_labels = next(csv.reader(attribution_file))

        # TODO: utilize appropriate encoding for categorical, non-numerical data
        df = pd.DataFrame(self.attributions, columns=self.feature_labels)
        #pd.get_dummies(obj_df, columns=["drive_wheels"]).head()

    @staticmethod
    def get_subpopulation_sizes(subpopulations):
        """Computes the sizes of the subpopulations using membership array
        Args:
            subpopulations (list): contains index of cluster each sample belongs to.
                Example, [0, 1, 0, 0].
        Returns:
            list: size of each subpopulation ordered by index. Example: [3, 1]
        """
        index_to_size = Counter(subpopulations)
        sizes = [index_to_size[i] for i in sorted(index_to_size)]

        return sizes

    @staticmethod
    def get_subpopulation_sizes_lol(n, subpopulations):
        """Computes the sizes of the subpopulations using membership array

        Let list of medoid indices = [medoidIndex_{1}, medoidIndex2_{2}, ..., medoidIndex_{n}],
        where medoidIndex_{i} represents a datapoint in the dataframe.

        Args:
            subpopulations (list of lists):
            a list of lists of cluster members
            where each list of cluster members contains row indices of cluster each sample belongs to.
            We assume the dataset comes unordered and but our gam.py assigns row indices anyway.
                Example, [[rowIndex_1, rowIndex_2, ..., rowIndex_1], ..., [..., ...,rowIndex_i]],
                where rowIndex_i < rowIndex_n = last row index of dataset.

        Returns:
            list: size of each subpopulation ordered by index.
            Example: [medoidIndex_{i_1}, medoidIndex2_{i_2}, medoidIndex_{i_n}]
            where medoidIndex_{i_n} <= medoidIndex_{n}.
            medoidIndex refers to the index in the array of final medoids calculated.
        """
        subpop = [None]*n # create empty array to fill with medoid indices
        for medoidIndex, members in enumerate(subpopulations):
            for rowIndex in members:
                subpop[rowIndex] = medoidIndex # assign medoidIndex for each rowIndex
        index_to_size = Counter(subpop)
        sizes = [index_to_size[i] for i in sorted(index_to_size)]

        return sizes

    def _get_explanations(self, centers):
        """Converts subpopulation centers into explanations using feature_labels

        Args:
            centers (list): index of subpopulation centers. Example: [21, 105, 3]

        Returns: explanations (list).
            Example: [[('height', 0.2), ('weight', 0.8)], [('height', 0.5), ('weight', 0.5)]].
        """
        explanations = []

        for center_index in centers:
            explanation_weights = self.clustering_attributions[center_index]
            explanations.append(list(zip(self.feature_labels, explanation_weights)))
        return explanations


    def plot(self, num_features=5, output_path_base=None, display=True):
        """Shows bar graph of feature importance per global explanation

        Args:
            num_features: number of top features to plot, int
            output_path_base: path to store plots
            display: option to display plot after generation, bool
        """
        if not hasattr(self, "explanations"):
            self.generate()

        fig_x, fig_y = 5, num_features

        for idx, explanations in enumerate(self.explanations):
            _, axs = plt.subplots(1, 1, figsize=(fig_x, fig_y), sharey=True)

            explanations_sorted = sorted(
                explanations, key=lambda x: x[-1], reverse=False
            )[-num_features:]
            axs.barh(*zip(*explanations_sorted))
            axs.set_xlim([0, 1])
            axs.set_title("Explanation {}".format(idx + 1), size=10)
            axs.set_xlabel("Importance", size=10)

            plt.tight_layout()
            if output_path_base:
                output_path = "{}_explanation_{}.png".format(output_path_base, idx + 1)
                # bbox_inches option prevents labels cutting off
                plt.savefig(output_path, bbox_inches="tight")

            if display:
                plt.show()

    def plot_explanations(self, gam_explanations, num_features=5, output_path_base=None, display=True):
        """Shows bar graph of feature importance per global explanation

        Args:
            num_features: number of top features to plot, int
            output_path_base: path to store plots
            display: option to display plot after generation, bool
        """

        fig_x, fig_y = 5, num_features

        for idx, explanations in enumerate(gam_explanations):
            _, axs = plt.subplots(1, 1, figsize=(fig_x, fig_y), sharey=True)

            explanations_sorted = sorted(
                explanations, key=lambda x: x[-1], reverse=False
            )[-num_features:]
            axs.barh(*zip(*explanations_sorted))
            axs.set_xlim([0, 1])
            axs.set_title("Explanation {}".format(idx + 1), size=10)
            axs.set_xlabel("Importance", size=10)

            plt.tight_layout()
            if output_path_base:
                output_path = "{}_explanation_{}.png".format(output_path_base, idx + 1)
                # bbox_inches option prevents labels cutting off
                plt.savefig(output_path, bbox_inches="tight")

            if display:
                plt.show()


    def generate(self, distance_function=None, max_iter=1000, tol=0.0001, parameters={}):
        """
        Clusters local attributions into subpopulations with global explanations
        """

        params = parameters
        self._read_local() # read in the data
        if self.use_normalized: # normalize clustering attributions
            self.clustering_attributions = GAM.normalize(self.attributions)
        else: # attributions non-normalized
            self.clustering_attributions = self.attributions
        print("self.attributions: ",str(self.attributions))


        # Cluster according to appropriate algorithm, distance metric
        # Calls local kmedoids module to group attributions
        # Use the distance metric and k-medoids algorithm specified
        # max_iter = maximum number of k-medoid updates
        # tol = minimum error for medoid optimum
        if self.cluster_method is None: # use regular k-medoids
            clusters = KMedoids(
                self.n_clusters,
                dist_func=self.distance_function,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            clusters.fit(self.clustering_attributions, verbose=False)

            self.subpopulations = clusters.members
            self.subpopulation_sizes = GAM.get_subpopulation_sizes(clusters.members)
            self.explanations = self._get_explanations(clusters.centers)

        elif self.cluster_method == "parallel medoids":
            clusters = ParallelMedoids()
            n, dfp, mlist, duration = clusters.fit(X=self.clustering_attributions, verbose=False, data=self.attributions_path)
            self.duration = duration
            self.subpopulations = clusters.members
            self.subpopulation_sizes = GAM.get_subpopulation_sizes_lol(n, clusters.members)
            self.explanations = self._get_explanations(clusters.centers)
            parallelPlot(dfp)
            radarPlot(dfp, mlist)
            facetedRadarPlot(dfp, mlist)
            silhouetteAnalysis(dfp, mlist)

        elif self.cluster_method == "ranked medoids":
            clusters = RankedMedoids()

            n, duration = clusters.fit(X=self.clustering_attributions, verbose=False, data = self.attributions_path)
            self.duration = duration
            self.subpopulations = clusters.members
            self.subpopulation_sizes = GAM.get_subpopulation_sizes_lol(n, clusters.members)
            self.explanations = self._get_explanations(clusters.centers)
            #parallelPlot(dfp)
            #radarPlot(dfp, mlist)
            #facetedRadarPlot(dfp, mlist)
        elif self.cluster_method == "spectralnet":

            #
            # SET UP INPUTS
            #

            # create true y placeholder (not used in unsupervised training)
            y_true = tf.placeholder(tf.float32, shape=(None, params['n_clusters']), name='y_true')

            batch_sizes = {
                'Unlabeled': params['batch_size'],
                'Labeled': params['batch_size'],
                'Orthonorm': params.get('batch_size_orthonorm', params['batch_size']),
            }

            input_shape = self.clustering_attributions.shape[1:]

            # spectralnet has three inputs -- they are defined here
            if params['isCustomData']:
                inputs = {
                    'Unlabeled': Input(shape=input_shape, name='UnlabeledInput'),
                    'Orthonorm': Input(shape=input_shape, name='OrthonormInput'),
                }
            else:
                inputs = {
                    'Unlabeled': Input(shape=input_shape, name='UnlabeledInput'),
                    'Labeled': Input(shape=input_shape, name='LabeledInput'),
                    'Orthonorm': Input(shape=input_shape, name='OrthonormInput'),
                }

            # #
            # # DEFINE AND TRAIN SIAMESE NET
            # #
            #
            # # run only if we are using a siamese network
            # if params['affinity'] == 'siamese':
            #     siamese_net = networks.SiameseNet(inputs, params['arch'], params.get('siam_reg'), y_true)
            #
            #     history = siamese_net.train(pairs_train, dist_train, pairs_val, dist_val,
            #                                 params['siam_lr'], params['siam_drop'], params['siam_patience'],
            #                                 params['siam_ne'], params['siam_batch_size'])
            #
            # else:
            #     siamese_net = None

            #
            # DEFINE AND TRAIN SPECTRALNET
            #

            # __init__(self, inputs=None, arch=None, spec_reg=None, y_true=None, y_train_labeled_onehot=None,
            #          n_clusters=None, affinity=None, scale_nbr=None, n_nbrs=None, batch_sizes=None,
            #          siamese_net=None, x_train=None, have_labeled=False)
            spectral_net = networks2.SpectralNet(inputs=inputs, arch=params['arch'],
                                                 spec_reg=params.get('spec_reg'),
                                                 n_clusters=params['n_clusters'], affinity=params['affinity'], scale_nbr=params['scale_nbr'],
                                                 n_nbrs=params['n_nbrs'], batch_sizes=batch_sizes, x_train=self.clustering_attributions, isUnsupervised=params['isCustomData'])

            x_train_unlabeled = self.clustering_attributions
            x = self.clustering_attributions

            spectral_net.train(x_train_unlabeled=x_train_unlabeled, lr=params['spec_lr'], drop=params['spec_drop'], patience=params['spec_patience'], num_epochs=params['spec_ne'], isUnsupervised=params['isCustomData'])

            print("finished training")

            # EVALUATE
            # get final embeddings
            x_spectralnet = spectral_net.predict(x)

            # get accuracy and nmi
            kmeans_assignments, km = get_cluster_sols(x_spectralnet, ClusterClass=KMeans, n_clusters=params['n_clusters'], init_args={'n_init': 10})

            ### only use for labeled data ###
            # y_spectralnet, _ = get_y_preds(kmeans_assignments, y, params['n_clusters'])
            # print_accuracy(kmeans_assignments, y, params['n_clusters'])
            #
            # from sklearn.metrics import normalized_mutual_info_score as nmi
            # nmi_score = nmi(kmeans_assignments, y)
            # print('NMI: ' + str(np.round(nmi_score, 3)))
            #
            # if params['generalization_metrics']:
            #     x_spectralnet_train = spectral_net.predict(x_train_unlabeled)
            #     x_spectralnet_test = spectral_net.predict(x_test)
            #     km_train = KMeans(n_clusters=params['n_clusters']).fit(x_spectralnet_train)
            #     from scipy.spatial.distance import cdist
            #     dist_mat = cdist(x_spectralnet_test, km_train.cluster_centers_)
            #     closest_cluster = np.argmin(dist_mat, axis=1)
            #     print_accuracy(closest_cluster, y_test, params['n_clusters'], ' generalization')
            #     nmi_score = nmi(closest_cluster, y_test)
            #     print('generalization NMI: ' + str(np.round(nmi_score, 3)))
            #
            # return x_spectralnet, y_spectralnet

            n, dfp, mlist, duration = spectral_net.fit(X=self.clustering_attributions, verbose=False, data=self.attributions_path)
            self.duration = duration
            self.subpopulations = spectral_net.members
            self.subpopulation_sizes = GAM.get_subpopulation_sizes_lol(n, spectral_net.members)
            self.explanations = self._get_explanations(spectral_net.centers)
            parallelPlot(dfp)
            radarPlot(dfp, mlist)
            facetedRadarPlot(dfp, mlist)
            silhouetteAnalysis(dfp, mlist)

        elif self.cluster_method == "banditPAM":
            banditPAM = BanditPAM()
            n, imgs, feature_labels, duration = banditPAM.fit(X=self.clustering_attributions, verbose=False, data = self.attributions_path)
            self.duration = duration

            self.clustering_attributions = imgs
            self.attributions = imgs
            self.feature_labels = feature_labels
            #self.feature_labels = range(1, len(imgs[0]+1))

            self.subpopulations = banditPAM.members
            self.subpopulation_sizes = GAM.get_subpopulation_sizes_lol(n, banditPAM.members)
            self.explanations = self._get_explanations(banditPAM.centers)
        else: # use passed in cluster_method and pass in GAM itself
            self.cluster_method(self)

        # Use scoring method if one is provided at initialization
        if self.scoring_method:
            self.score = self.scoring_method(self)

if __name__ == '__main__':

    # PARSE ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
    parser.add_argument('--dset', type=str, help='gpu number to use', default='mnist')
    args = parser.parse_args()

    args.dset = 'custom'

    # SELECT GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    params = defaultdict(lambda: None)

    # SET GENERAL HYPERPARAMETERS
    general_params = {
        'dset': args.dset,  # dataset: reuters / mnist
        'val_set_fraction': 0.1,  # fraction of training set to use as validation
        'precomputedKNNPath': '',
        # path for precomputed nearest neighbors (with indices and saved as a pickle or h5py file)
        'siam_batch_size': 128,  # minibatch size for siamese net
    }
    params.update(general_params)

    # SET DATASET SPECIFIC HYPERPARAMETERS
    if args.dset == 'mnist':
        mnist_params = {
            'n_clusters': 10,  # number of clusters in data
            'use_code_space': True,  # enable / disable code space embedding
            'affinity': 'siamese',  # affinity type: siamese / knn
            'n_nbrs': 3,  # number of nonzero entries (neighbors) to use for graph Laplacian affinity matrix
            'scale_nbr': 2,  # neighbor used to determine scale of gaussian graph Laplacian; calculated by
            # taking median distance of the (scale_nbr)th neighbor, over a set of size batch_size
            # sampled from the datset

            'siam_k': 2,  # threshold where, for all k <= siam_k closest neighbors to x_i, (x_i, k) is considered
            # a 'positive' pair by siamese net

            'siam_ne': 400,  # number of training epochs for siamese net
            'spec_ne': 400,  # number of training epochs for spectral net
            'siam_lr': 1e-3,  # initial learning rate for siamese net
            'spec_lr': 1e-3,  # initial learning rate for spectral net
            'siam_patience': 10,  # early stopping patience for siamese net
            'spec_patience': 20,  # early stopping patience for spectral net
            'siam_drop': 0.1,  # learning rate scheduler decay for siamese net
            'spec_drop': 0.1,  # learning rate scheduler decay for spectral net
            'batch_size': 1024,  # batch size for spectral net
            'siam_reg': None,  # regularization parameter for siamese net
            'spec_reg': None,  # regularization parameter for spectral net
            'siam_n': None,  # subset of the dataset used to construct training pairs for siamese net
            'siamese_tot_pairs': 600000,  # total number of pairs for siamese net
            'arch': [  # network architecture. if different architectures are desired for siamese net and
                #   spectral net, 'siam_arch' and 'spec_arch' keys can be used
                {'type': 'relu', 'size': 1024},
                {'type': 'relu', 'size': 1024},
                {'type': 'relu', 'size': 512},
                {'type': 'relu', 'size': 10},
            ],
            'use_approx': False,  # enable / disable approximate nearest neighbors
            'use_all_data': True,  # enable to use all data for training (no test set)
        }
        params.update(mnist_params)
    elif args.dset == 'reuters':
        reuters_params = {
            'n_clusters': 4,
            'use_code_space': True,
            'affinity': 'siamese',
            'n_nbrs': 30,
            'scale_nbr': 10,
            'siam_k': 100,
            'siam_ne': 20,
            'spec_ne': 300,
            'siam_lr': 1e-3,
            'spec_lr': 5e-5,
            'siam_patience': 1,
            'spec_patience': 5,
            'siam_drop': 0.1,
            'spec_drop': 0.1,
            'batch_size': 2048,
            'siam_reg': 1e-2,
            'spec_reg': 5e-1,
            'siam_n': None,
            'siamese_tot_pairs': 400000,
            'arch': [
                {'type': 'relu', 'size': 512},
                {'type': 'relu', 'size': 256},
                {'type': 'relu', 'size': 128},
            ],
            'use_approx': True,
            'use_all_data': True,
        }
        params.update(reuters_params)
    elif args.dset == 'cc':
        cc_params = {
            # data generation parameters
            'train_set_fraction': 1.,  # fraction of the dataset to use for training
            'noise_sig': 0.1,  # variance of the gaussian noise applied to x
            'n': 1500,  # number of total points in dataset
            # training parameters
            'n_clusters': 2,
            'use_code_space': False,
            'affinity': 'full',
            'n_nbrs': 2,
            'scale_nbr': 2,
            'spec_ne': 300,
            'spec_lr': 1e-3,
            'spec_patience': 30,
            'spec_drop': 0.1,
            'batch_size': 128,
            'batch_size_orthonorm': 128,
            'spec_reg': None,
            'arch': [
                {'type': 'softplus', 'size': 50},
                {'type': 'BatchNormalization'},
                {'type': 'softplus', 'size': 50},
                {'type': 'BatchNormalization'},
                {'type': 'softplus', 'size': 50},
                {'type': 'BatchNormalization'},
            ],
            'use_all_data': True,
        }
        params.update(cc_params)
    elif args.dset == 'cc_semisup':
        cc_semisup_params = {
            'dset': 'cc',  # dataset affects data loading in get_data() so we must set back to 'cc'
            # data generation parameters
            'train_set_fraction': .8,
            'noise_sig': 0.175,
            'n': 1900,
            # training parameters
            'train_labeled_fraction': 0.02,
            # fraction of the training set to provide labels for (in semisupervised experiments)
            'n_clusters': 2,
            'use_code_space': False,
            'affinity': 'full',
            'n_nbrs': 2,
            'scale_nbr': 2,
            'spec_ne': 300,
            'spec_lr': 1e-3,
            'spec_patience': 30,
            'spec_drop': 0.1,
            'batch_size': 128,
            'batch_size_orthonorm': 256,
            'spec_reg': None,
            'arch': [
                {'type': 'softplus', 'size': 50},
                {'type': 'BatchNormalization'},
                {'type': 'softplus', 'size': 50},
                {'type': 'BatchNormalization'},
                {'type': 'softplus', 'size': 50},
                {'type': 'BatchNormalization'},
            ],
            'generalization_metrics': True,  # enable to check out of set generalization error and nmi
            'use_all_data': False,
        }
        params.update(cc_semisup_params)
    elif args.dset == 'custom':
        cc_params = {
            # data generation parameters
            'train_set_fraction': 1.,  # fraction of the dataset to use for training
            'noise_sig': 0.1,  # variance of the gaussian noise applied to x
            'n': 1500,  # number of total points in dataset
            # training parameters
            'n_clusters': 2,
            'use_code_space': False,
            'affinity': 'full',
            'n_nbrs': 2,
            'scale_nbr': 2,
            'spec_ne': 2,
            'spec_lr': 1e-3,
            'spec_patience': 30,
            'spec_drop': 0.1,
            'batch_size': 128,
            'batch_size_orthonorm': 128,
            'spec_reg': None,
            'arch': [
                {'type': 'softplus', 'size': 50},
                {'type': 'BatchNormalization'},
                {'type': 'softplus', 'size': 50},
                {'type': 'BatchNormalization'},
                {'type': 'softplus', 'size': 50},
                {'type': 'BatchNormalization'},
            ],
            'use_all_data': True,
            'isCustomData': True
        }
        params.update(cc_params)

    # params['n_clusters'] = 3
    # params['spec_ne'] = 10 # number of epochs
    #
    # # LOAD DATA
    # data = get_data(params)
    #
    # # RUN EXPERIMENT
    # x_spectralnet, y_spectralnet = run_net(data, params)
    #
    # if args.dset in ['cc', 'cc_semisup']:
    #     # run plotting script
    #     import plot_2d
    #
    #     plot_2d.process(x_spectralnet, y_spectralnet, data, params)

    ################################################################################################################
    #local_attribution_path = 'data/mushroom-attributions-200-samples.csv' # the pathway to the data file
    local_attribution_path = 'data/mushroom-attributions-200-samples.csv'
    g = GAM(attributions_path = local_attribution_path, n_clusters=3, cluster_method='spectralnet') # initialize GAM with filename, k=number of clusters
    g.generate(parameters=params) # generate GAM using k-medoids algorithm with number of features specified
    g.plot(num_features=7) # plot the GAM
    g.subpopulation_sizes # generate subpopulation sizes
    g.explanations # generate explanations