"""
kernel_medodids contains the class KernelMedoids which is used to run the kernel algorithm based on the paper:
    Randomized Clustered Nystrom for Large-Scale Kernel Machines

KernelMedoids can be initialized from GAM with the desired arguments

The fit function is called from GAM

The fit function calls cluster which is then completes the kernel algorithm

The algorithm utilizes nystroms and the vaex framework to improve efficiency

After the kernel algorithm is complete, it needs to call another clustering algorithm to complete the clustering

Currently that algorithm is set to be the parallel medoids algorithm

"""

import sklearn
import vaex
from gam_package.preprocessor.preprocessor import load_data, setArguments
import csv
import logging
from collections import Counter

import matplotlib.pylab as plt
import numpy as np
from numpy import sqrt, sum, matmul
from math import exp
from scipy.sparse.linalg import svds
from timeit import default_timer
import math
from numpy.linalg import svd

from gam_package.medoids_algorithms.k_medoids import KMedoids
from gam_package.distance_functions.kendall_tau_distance import mergeSortDistance
from gam_package.distance_functions.spearman_distance import spearman_squared_distance
from gam_package.distance_functions.euclidean_distance import euclidean_distance

from gam_package.medoids_algorithms.parallel_medoids import ParallelMedoids
from gam_package.plot_functions.plot import parallelPlot, radarPlot, facetedRadarPlot, silhouetteAnalysis
from gam_package.medoids_algorithms.ranked_medoids import RankedMedoids
from gam_package.medoids_algorithms.bandit_pam import BanditPAM

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans


class KernelMedoids:

    def __init__(self, n_clusters=1, max_iter=100, tol=0.0001, attributions_path="data/mushrooms.csv",
                 CLUSTER_NUM=3, TARGET_DIM=6, SKETCH_SIZE=60, SIGMA=1, dataset=None):

        self.attributions_path = attributions_path
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centers = None
        self.members = None

        ## Parse parameters from command line arguments
        self.CLUSTER_NUM = CLUSTER_NUM
        self.TARGET_DIM = CLUSTER_NUM * 2
        self.SKETCH_SIZE = TARGET_DIM * 10
        self.SIGMA = SIGMA
        self.dataset = dataset

        print("####################################")
        print("cluster number = ", CLUSTER_NUM)
        print("target dimension = ", TARGET_DIM)
        print("sketch size = ", SKETCH_SIZE)
        print("sigma = ", SIGMA)

        # import os
        # cwd = os.getcwd()  # Get the current working directory (cwd)
        # files = os.listdir(cwd)  # Get all the files in that directory
        # print("Files in %r: %s" % (cwd, files))

        self.applyOneOverSquareRoot_vfunc = np.vectorize(self.applyOneOverSquareRoot)

    def fit(self, X=None, plotit=False, verbose=True, attributions_path=None):

        ## Loads data
        # self.data = vaex.from_csv(self.attributions_path, copy_index = True)

        self.args = setArguments(self.dataset)
        total_data, total_labels, sigma, feature_labels = load_data(self.args)
        # total_data = total_data[np.random.choice(range(len(total_data)), size=args.sample_size, replace=False)]

        ## Parse the data to get labels and features
        # RDD[(Int, Array[Double])]
        # label_vector_rdd = df.rdd.map(pair= > (pair[0], pair[1]) )
        self.total_data = total_data
        x = total_data
        self.sigma = sigma
        self.feature_labels = feature_labels

        if total_labels is None:
            y = np.array(range(x.shape[0]))
        else:
            y = np.array(total_labels)
        label_vector_rdd = vaex.from_arrays(x=x, y=y)

        self.data = (total_data, total_labels, sigma, feature_labels)

        ## Perform kernel k-means with Nystrom approximation
        ## (Array[String], Array[String])
        self.cluster(label_vector_rdd, self.CLUSTER_NUM, self.TARGET_DIM, self.SKETCH_SIZE, self.SIGMA)

        return self.n, self.total_data, self.feature_labels, self.duration
        print("")

    '''
     * Compute the RBF kernel function of two vectors.
     def rbf(x1: Array[Double], x2: Array[Double], sigma: Double): Double
     '''

    def rbf(self, x1, x2, sigma):
        ## squared l2 distance between x1 and x2
        # dist = zip(x1, x2).map(lambda pair: pair[0] - pair[1]).map(lambda x: x * x).sum
        dist = zip(x1, x2)
        dist2 = list(map(lambda pair: pair[0] - pair[1], dist))
        dist3 = list(map(lambda x: x * x, dist2))
        sum_distance = sum(dist3)

        ## RBF kernel function is exp( - ||x1-x2||^2 / 2 / sigma^2 )
        return exp(-1 * sum_distance / (2 * sigma * sigma))

    '''
         * The Nystrom method for approximating the RBF kernel matrix.
         * Let K be n-by-n kernel matrix. The Nystrom method approximates
         * K by C * W^{-1}_l * C^T, where we set l = c/2.
         * The Nystrom feature vectors are the rows of C * W^{-1/2}_l.
         *
         * Input
         *  label_vector_rdd: RDD of labels and raw input feature vectors
         *  c: sketch size (k < s < c/2)
         *  sigma: kernel width parameter
         *
         * Output
         *  RDD of (lable, Nystrom feature vector) pairs
         *  the dimension of Nystrom feature vector is (c/
         return RDD[(Int, Vector)]
         '''

    def applyOneOverSquareRoot(self, s):
        return 1 / math.sqrt(s)

    '''
         * Locally compute the W matrix of Nystrom and its factorization W_{l}^{-1} = U * U^T.
         def nystrom_w_mat(data_samples: Array[Array[Double]], sigma: Double): DenseMatrix[Double]
         '''

    def nystrom_w_mat(self, data_samples, sigma):
        c = data_samples.shape[0]

        l = math.ceil(c * 0.5)

        ## Compute the W matrix of Nystrom method
        # w_mat = DenseMatrix.zeros[Double](c, c)
        w_mat = np.array([[0] * c] * c)
        # data_samplesX = data_samples.evaluate(data_samples.x)

        for j in range(c):
            for i in range(c):
                w_mat[i, j] = self.rbf(data_samples[i], data_samples[j], sigma)

        ## Compute matrix U such that U * U^T = W_{l}^{-1}
        U, S, V = svd(w_mat, full_matrices=False)

        # u_mat = U(::, 0 until l) # get matrix where only the first l values of every column is returned
        # u_mat = U[0:l,:]
        u_mat = U[:, 0:l]

        # s_arr = S(0 until l).toArray.map(lambda s: 1 / math.sqrt(s))
        s_arr = S[0:l]

        # s_arr.map(lambda s: 1 / math.sqrt(s))

        # s_arr = self.applyOneOverSquareRoot(s_arr)
        s_arr = self.applyOneOverSquareRoot_vfunc(s_arr)

        for j in range(0, l):
            # u_mat(::, j) *= s_arr(j)
            u_mat[:, j] = u_mat[:, j] * s_arr[j]
        return u_mat

    '''
     * Compute the RBF kernel functions of a vector and a collection of vectors.
     def rbf(x1: Array[Double], x2: Array[Array[Double]], sigma: Double): Array[Double]
     '''

    def rbf2(self, x1, x2, sigma):
        n = x2.shape[0]
        # print("n: ", n)
        kernel_arr = [0] * n
        # print("kernel_arr: ", kernel_arr)
        sigma_sq = 2 * sigma * sigma
        # print("sigma_sq: ", sigma_sq)
        for i in range(0, n):
            ## squared l2 distance between x1 and x2(i)
            # dist = zip(x1,x2[i])
            # dist2 = list(map(lambda pair: pair[0] - pair[1], dist))
            # dist3 = list(map(lambda x: x * x, dist2))
            # sum_distance = sum(dist3)
            # print("i: ", i)
            sum_distance = sqrt(sum((x1 - x2[i]) ** 2))
            # print("sum_distance: ", sum_distance)
            ## RBF kernel function
            kernel_arr[i] = exp(-sum_distance / sigma_sq)
            # print("kernel_arr[i]: ", kernel_arr[i])
        # print("kernel_arr: ", kernel_arr)
        # print("kernel_arr.shape: ", kernel_arr.shape)
        return kernel_arr

    def applyRBF2(self, row):
        # print("line 207, type(row): ", type(row))
        return self.rbf2(row, self.data_samples, self.sigma)

    def nystrom(self, label_vector_rdd, c, sigma):
        n = label_vector_rdd.shape[0]
        self.n = n

        ## Randomly sample about c points from the dataset
        frac = c / n
        # data_samples = label_vector_rdd.sample(false, frac).map(pair= > pair._2).collect
        # broadcast_samples = sc.broadcast(data_samples)
        # RDD[(Int, Array[Double])]
        data_samples = label_vector_rdd.sample(frac=frac)
        # data_samples = data_samples.x
        data_samples = data_samples.evaluate(data_samples.x)

        ## Compute the C matrix of the Nystrom method
        # rbf_fun = (x1: Array[Double], x2: Array[Array[Double]])

        # c_mat_rdd = label_vector_rdd
        # .map(pair => (pair._1, broadcast_rbf.value(pair._2, broadcast_samples.value)))
        # .map(pair => (pair._1, new DenseVector(pair._2)))

        self.data_samples = data_samples
        self.sigma = sigma
        c_mat_rddX = label_vector_rdd.apply(self.applyRBF2, arguments=[label_vector_rdd.x])

        # c_mat_rddX = label_vector_rdd.evaluate(c_mat_rddX)

        x = label_vector_rdd.evaluate(c_mat_rddX)
        y = label_vector_rdd.evaluate(label_vector_rdd.y)
        c_mat_rdd = vaex.from_arrays(x=x, y=y)

        ## Compute the W matrix of the Nystrom method
        ## decompose W as: U * U^T \approx W^{-1}
        u_mat = self.nystrom_w_mat(data_samples, sigma)

        print("")
        ## Compute the features extracted by Nystrom
        ## feature matrix is C * W^{-1/2}_{c/2}
        # nystrom_rdd = c_mat_rdd.map(lambda row: row.t * u_mat).map(lambda row: pair._2.t)
        print("")
        self.u_mat = u_mat
        nystrom_rddX = c_mat_rdd.apply(self.multiplyRowUMat, arguments=[c_mat_rdd.x])  # returned only x values

        x2 = c_mat_rdd.evaluate(nystrom_rddX)
        nystrom_rdd = vaex.from_arrays(x=x2, y=y)

        return nystrom_rdd

    def multiplyRowUMat(self, row):
        retVal = matmul(row, self.u_mat)
        # print("retVal: ", retVal)
        return retVal

    def multiplyRowVMat(self, row):
        retVal = matmul(self.v_mat, row)
        # print("multiplyRowVMat: ", retVal)
        return retVal

    '''
     Kernel k-means clustering with Nystrom approximation.
     Input
     *  label_vector_rdd: RDD of labels and raw input feature vectors
     *  k: cluster number
     *  s: target dimension of extracted feature vectors
     *  c: sketch size (k < s < c/2)
     *  sigma: kernel width parameter
     *
     * Output
     *  labels: Array of (true label, predicted label) pairs
     *  time: Array of the elapsed time of Nystrom, PCA, and k-means, respectively
     return (Array[String], Array[String])
    '''

    def cluster(self, label_vector_rdd, k, s, c, sigma):
        ## max number of iterations (can be tuned)
        MAX_ITER = 100

        # label_vector_rdd

        ## Extract features by the Nystrom method
        t0 = default_timer()

        # nystrom_rdd: RDD[(Int, Vector)]
        nystrom_rdd = self.nystrom(label_vector_rdd, c, sigma)
        t1 = default_timer() - t0

        ##label_vector_rdd.unpersist()
        print("####################################")
        print("Nystrom method costs  ", t1, "  seconds.")
        print("####################################")

        ## Extract s principal components from the Nystrom features
        ## The V matrix stored in a local dense matrix
        t2 = default_timer()
        # mat = new RowMatrix(nystrom_rdd.map(pair => pair._2))
        mat = nystrom_rdd.evaluate(nystrom_rdd.x)

        # svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(s, computeU = false)
        U, S, Vt = svds(mat, k=s)
        print("")
        # v_mat: Matrix = svd.V.transpose
        # v_mat = V.transpose
        v_mat = Vt

        # nystrom_pca_rdd: RDD[(Int, Vector)] = nystrom_rdd
        #         .map(pair => (pair._1, broadcast_v_mat.value.multiply(pair._2)))
        #         .map(pair => (pair._1, Vectors.dense(pair._2.toArray)))

        self.v_mat = v_mat

        # nystrom_pca_rdd = nystrom_rdd.apply(, arguments[])

        nystrom_pca_rddX = nystrom_rdd.apply(self.multiplyRowVMat, arguments=[nystrom_rdd.x])

        x = nystrom_rdd.evaluate(nystrom_pca_rddX)
        # x = nystrom_pca_rddX
        y = nystrom_rdd.evaluate(nystrom_rdd.y)
        nystrom_pca_rdd = vaex.from_arrays(x=x, y=y)
        t3 = default_timer() - t2

        print("####################################")
        print("PCA costs  ", t3, "  seconds.")
        print("####################################")

        print('The scikit-learn version is {}.'.format(sklearn.__version__))

        ## K-means clustering over the extracted features
        t4 = default_timer()
        # feature_rdd: RDD[Vector] = nystrom_pca_rdd.map(pair => pair._2)
        feature_rdd = x
        # clusters = KMeans.train(feature_rdd, k, MAX_ITER)

        # kmedoids = KMedoids(n_clusters=k, max_iter=self.max_iter)
        # centers, members, duration = kmedoids.fit(feature_rdd)
        # banditPAM = BanditPAM(data=self.data)
        # n, total_data, feature_labels, duration = banditPAM.fit(X=feature_rdd, verbose=False)

        clusters = ParallelMedoids(attributions_path=self.args.attributions_path)
        n, dfp, mlist, duration = clusters.fit(X=feature_rdd, verbose=False, n_clusters=k)

        self.centers = clusters.centers
        self.members = clusters.members

        t5 = default_timer() - t4
        self.duration = default_timer() - t0
        print("####################################")
        print("K-means clustering costs  ", t5, "  seconds.")
        print("####################################")

        ## Predict labels
        # labels: Array[String] = nystrom_pca_rdd
        #         .map(pair => (pair._1, clusters.predict(pair._2)))
        #         .map(pair => pair._1.toString + " " + pair._2.toString)
        #         .collect()

        # return (labels, time)

    # /mnt/c/Users/charm/PycharmProjects/SparkKernelKMeans/data


# '''

if __name__ == '__main__':
    kernelMedoids = KernelMedoids(max_iter=1, dataset="mushrooms")
    n, total_data, feature_labels, duration = kernelMedoids.fit()
    print("centers: ", kernelMedoids.centers)
    print("members: ", kernelMedoids.members)