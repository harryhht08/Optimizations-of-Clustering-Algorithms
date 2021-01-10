import csv
import os
import numpy as np
import mnist
import argparse
import pandas as pd
import pickle
import shlex

import csv
import logging
from collections import Counter

import matplotlib.pylab as plt
import numpy as np

from gam_package.medoids_algorithms.k_medoids import KMedoids
from gam_package.distance_functions.kendall_tau_distance import mergeSortDistance
from gam_package.distance_functions.spearman_distance import spearman_squared_distance
from gam_package.distance_functions.euclidean_distance import euclidean_distance

from gam_package.medoids_algorithms.parallel_medoids import ParallelMedoids
from gam_package.plot_functions.plot import parallelPlot, radarPlot, facetedRadarPlot, silhouetteAnalysis
from gam_package.medoids_algorithms.ranked_medoids import RankedMedoids

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def setArguments(datasetFilePath, num_samp=200, n_clusters = 3):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', help='print debugging output', action='count', default=0, required=False)
    parser.add_argument('-k', '--num_medoids', help='Number of medoids', type=int, default=3, required=False)
    parser.add_argument('-N', '--sample_size', help='Sampling size of dataset', type=int, default=num_samp,
                        required=False)
    parser.add_argument('-s', '--seed', help='Random seed', type=int, default=42, required=False)
    parser.add_argument('-d', '--dataset', help='Dataset to use', type=str, default='MNIST', required=False)
    parser.add_argument('-c', '--cache_computed', help='Cache computed', default=None, required=False)
    parser.add_argument('-m', '--metric', help='Metric to use (L1 or L2)', type=str, required=False)
    parser.add_argument('-f', '--force', help='Recompute Experiments', action='store_true', required=False)
    parser.add_argument('-p', '--fast_pam1', help='Use FastPAM1 optimization', action='store_true', required=False)
    parser.add_argument('-r', '--fast_pam2', help='Use FastPAM2 optimization', action='store_true', required=False)
    parser.add_argument('-w', '--warm_start_medoids', help='Initial medoids to start with', type=str, default='',
                        required=False)
    parser.add_argument('-B', '--build_ao_swap', help='Build or Swap, B = just build, S = just swap, BS = both',
                        type=str, default='BS', required=False)
    parser.add_argument('-e', '--exp_config', help='Experiment configuration file to use', required=False)

    cmdline = "-k 5 -N 1000 -s 42 -d MNIST -m L2 -p"
    args = parser.parse_args(shlex.split(cmdline))


    args.dataset = datasetFilePath
    args.metric = 'L2'
    args.fast_pam1 = True
    args.num_medoids = n_clusters

    cwd = os.getcwd()  # Get the current working directory (cwd)
    files = os.listdir(cwd)  # Get all the files in that directory
    print("Files in %r: %s" % (cwd, files))

    if args.dataset == 'MNIST':
        pass

    elif args.dataset == 'mushrooms':
        args.sample_size = num_samp # num_samp
        args.attributions_path = "data/mushrooms.csv"
    elif args.dataset == 'wine':
        args.sample_size = num_samp
        args.attributions_path = "data/wine.csv"
    elif args.dataset == 'mice_protein':
        args.sample_size = num_samp
        args.attributions_path = "data/mice_protein.csv"
    elif args.dataset == 'crime':
        args.sample_size = num_samp
        args.attributions_path = "data/crime_without_states.csv"
    else:
        # raise Exception("Didn't specify a valid dataset")
        print("")

    return args

def load_data(args):
    print("data_utils -> load_data")
    '''
    Load the different datasets, as a numpy matrix if possible. In the case of
    HOC4, load the datasets as a list of trees.
    '''

    cwd = os.getcwd()  # Get the current working directory (cwd)
    files = os.listdir(cwd)  # Get all the files in that directory
    print("Files in %r: %s" % (cwd, files))

    #cwdSplit = cwd.split("\\")
    cwdSplit = cwd.split("/")
    currentFolder = cwdSplit[-1]
    prependDoubleDots = True
    if currentFolder == 'gam_package':
        prependDoubleDots = False

    filepath = args.attributions_path
    if prependDoubleDots:
        filepath = "../" + filepath

    if args.dataset == 'MNIST':
        N = 70000
        m = 28
        sigma = 0.7
        train_data = mnist.train_images()
        train_labels = mnist.train_labels()
        test_data = mnist.test_images()
        test_labels = mnist.test_labels()
        total_data = np.append(train_data, test_data, axis=0)
        total_labels = np.append(train_labels, test_labels, axis=0)
        feature_labels = range(784)
        return total_data.reshape(N, m * m) / 255, total_labels, sigma, feature_labels

    elif args.dataset == 'mushrooms':

        total_data = np.genfromtxt(filepath, dtype=float, delimiter=",", skip_header=1)
        with open(filepath) as attribution_file:
            feature_labels = next(csv.reader(attribution_file))
        sigma = 0.01
        # unsupervised, so no total_labels
        return total_data, None, sigma, feature_labels

    elif args.dataset == 'wine':
        total_data = np.genfromtxt(filepath, dtype=float, delimiter=",", skip_header=1)
        with open(filepath) as attribution_file:
            feature_labels = next(csv.reader(attribution_file))
        sigma = 0.01
        return total_data, None, sigma, feature_labels

    elif args.dataset == 'mice_protein':

        total_data = np.genfromtxt(filepath, dtype=float, delimiter=",", skip_header=1)
        with open(filepath) as attribution_file:
            feature_labels = next(csv.reader(attribution_file))
        sigma = 0.01
        total_data = np.nan_to_num(total_data)
        return total_data, None, sigma, feature_labels

    elif args.dataset == 'crime':
        total_data = np.genfromtxt(filepath, dtype=float, delimiter=",", skip_header=1)
        with open(filepath) as attribution_file:
            feature_labels = next(csv.reader(attribution_file))
        sigma = 0.01
        total_data = np.nan_to_num(total_data)
        return total_data, None, sigma, feature_labels

    else:
        raise Exception("Didn't specify a valid dataset")