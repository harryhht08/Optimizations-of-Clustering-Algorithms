# import csv
# import vaex
#
# from pyspark import SparkContext
# from pyspark.sql import SQLContext
# import pandas as pd
# import numpy as np
# import math
# from scipy.spatial.distance import pdist, squareform
# import copy
# import multiprocessing as mp
# import matplotlib.pyplot as plt
# from timeit import default_timer
#
# class ParallelMedoids2:
#     """
#     Generates medoids through multiprocessing
#
#     Args:
#         n_clusters = desired number of clusters, default 1
#         dist_func = distance metric used for calculating distance, default euclidean
#         max_iter = max number of iterations for updating medoids, default 1000
#         tol = stopping critera for updating medoids, default 0.0001
#
#     """
#
#     def __init__(self, n_clusters=1, dist_func='euclidean', max_iter=1000, tol=0.0001,
#                  attributions_path="data/mushrooms.csv"):
#         self.attributions_path = attributions_path
#         self.k = n_clusters
#         if dist_func == 'euclidean':
#             self.dist = self.distance
#         self.max_iter = max_iter
#         self.tol = tol
#         self.centers = None
#         self.members = None
#
#     # creates a vector with distances from the current sample to each medoid
#     # returns the smallest distance
#     def distance(self, row):
#         vector = []
#         row_df = row.to_frame()
#         row_df = row_df.transpose()
#
#         # loops through each medoid and calculates the distance from the current sample
#         for m in medoidsList:
#             sum = 0
#             # calculates euclidean distance
#             for i in range(0, len(m.values[0])):
#                 m_value = m.values[0][i]
#                 row_df_value = row_df.values[0][i]
#                 sum += math.sqrt((m_value - row_df_value) ** 2)
#             vector.append(sum)
#         return min(vector)
#
#
#     # chooses the first selection of medoids using a k-means++ style algorithm
#     # uses a probability distribution to select better initial medoids than randomly choosing them
#     def initialSeeds(self, dfx, k):
#       N = dfx.count()
#
#       #dfp = df.toPandas()
#       sample = dfx.sample()
#       # global medoidsList
#       medoidsList = []
#
#       # chooses the first medoid at random
#       medoidsList.append(sample)
#
#       # uses k-means++ to choose the remaining initial medoids
#       for iter in range(1,k):
#
#         # uses multiprocessing to speed up the apply function on distance
#         distances_series = dfx.apply(self.distance, arguments=[dfx.x])
#
#         # distancesdf = distances_series.to_frame()
#
#         distances_array = np.concatenate(distances_series.values, axis = 0)
#         sum = distancesdf[0].sum()
#
#         # create probability distribution
#         distances_array = [distance / sum for distance in distances_array]
#
#         # find new medoid based on probability distribution
#         newMedoidId = np.random.choice(N, 1, p = distances_array)
#         newMedoid = dfp.iloc[newMedoidId[0]]
#         newMedoid = newMedoid.to_frame().transpose()
#         medoidsList.append(newMedoid)
#       return medoidsList
#
#     # returns the index of the nearest medoid for the given sample
#     def nearestCluster(self, row):
#       nearestClustersMap = {}
#       row_df = row.to_frame()
#       row_df = row_df.transpose()
#       for m in medoidsList:
#         sum = 0
#         for i in range(0, len(m.values[0])):
#             m_value = m.values[0][i]
#             row_df_value = row_df.values[0][i]
#             sum += math.sqrt((m_value - row_df_value)**2)
#         nearestClustersMap[m.index[0]] = sum
#
#       return min(nearestClustersMap, key=nearestClustersMap.get)
#
#     # finds a new medoid given a cluster of samples
#     def exactMedoidUpdate(self, patternsInClusters):
#         # turn the patterns into a matrix
#         patterns = np.asmatrix(patternsInClusters)
#         # calculate the distance matrix using the matrix patterns
#         distanceMatrix = pdist(patterns, 'euclidean')
#         # sum across all the rows
#         sumRows = squareform(distanceMatrix).sum(axis = 1)
#         # take the medoid with the smallest sum
#         minIndex = np.argmin(sumRows)
#         newMedoid = patternsInClusters.iloc[minIndex].to_frame().transpose()
#         return newMedoid
#
#     def fit(self, X = None, plotit=False, verbose=True, n_clusters = 3):
#         """
#         Fits kmedoids with the option for plotting
#         """
#         start = default_timer()
#         _,_, n, dfp, mlist = self._cluster(self.attributions_path, n_clusters)
#         duration = default_timer() - start
#
#         if plotit:
#             _, ax = plt.subplots(1, 1)
#             colors = ["b", "g", "r", "c", "m", "y", "k"]
#             if self.k > len(colors):
#                 raise ValueError("we need more colors")
#
#             for i in range(len(self.centers)):
#                 X_c = X[self.members == i, :]
#                 ax.scatter(X_c[:, 0], X_c[:, 1], c=colors[i], alpha=0.5, s=30)
#                 ax.scatter(
#                     X[self.centers[i], 0],
#                     X[self.centers[i], 1],
#                     c=colors[i],
#                     alpha=1.0,
#                     s=250,
#                     marker="*",
#                 )
#         return n, dfp, mlist, duration
#
#     def _cluster(self, attributions_path, n_clusters):
#
#         #############################################################
#         # use numpy to process data from csv file, the header labels are discarded
#         self.attributions = np.genfromtxt(
#             self.attributions_path, dtype=float, delimiter=",", skip_header=1
#         )
#         # extract the feature labels
#         with open(self.attributions_path) as attribution_file:
#             self.feature_labels = next(csv.reader(attribution_file))
#
#         dataframe = pd.DataFrame(self.attributions, columns=self.feature_labels)
#         if dataframe.isnull().values.any():
#             self.df = dataframe.fillna(dataframe.mean())
#             self.attributions = self.df.values
#         else:
#             self.df = dataframe
#
#         x = dataframe.to_numpy()
#         y = np.array(range(x.shape[0]))
#         dfx = vaex.from_arrays(x=x, y=y)
#
#         max_iter = 10; k = n_clusters; sumOfDistances1 = float("inf")
#         #df = df.na.drop()
#         # if df.isnull().values.any():
#         #     df = df.fillna(df.mean())
#         # df_filled = df.fillna(value=0)
#
#         # set medoids equal to the initial medoids
#         medoids = self.initialSeeds(dfx, k)
#         global nearestClustersGlobal
#         # dfp = df.toPandas()
#
#         # updates medoids until criteria is reached or the maximum number of iterations is reached
#         for iter in range(0, max_iter):
#             previousMedoids = copy.deepcopy(medoids)
#
#             # calculates nearest distances and nearest clusters using multiprocessing
#             distances_series = self.apply_by_multiprocessing(dfp, self.distance, axis=1, workers=4)
#             nearestDistancesPDF = distances_series.to_frame()
#
#             nearest_clusters = self.apply_by_multiprocessing(dfp, self.nearestCluster, axis=1, workers=4)
#             nearestClustersPDF = nearest_clusters.to_frame()
#             nearestClustersGlobal = nearestClustersPDF
#
#             # updates each medoid for the given cluster
#             for mindex, m in enumerate(medoids):
#                 clID = m.index[0]
#
#                 # creates patternsIndex, an array where if row i is in cluster j, patternsIndex[i] = j
#                 patternsIndex = []
#                 nearestClustersPDF_flat = nearestClustersPDF.values.flatten()
#                 for i in range(len(nearestClustersPDF_flat)):
#                     if nearestClustersPDF_flat[i] == clID:
#                         patternsIndex.append(i)
#                 patternsInClusterPDF = dfp.iloc[patternsIndex, :]
#
#                 # updates the medoid for the current cluster
#                 newMedoid = self.exactMedoidUpdate(patternsInClusterPDF)
#                 medoids[mindex] = newMedoid
#             sumOfDistances2 = nearestDistancesPDF.sum(axis=0)[0]
#
#             # if the stopping criteria is met, stop updating medoids
#             if abs(sumOfDistances1 - sumOfDistances2) == 0:
#                 medoids = previousMedoids
#                 break
#             else:
#                 sumOfDistances1 = sumOfDistances2
#
#         # groupsDict is a dictionary with the (key:value) as (medoid:list of samples in cluster)
#         groupsDict = {}
#         for i in range(len(medoidsList)):
#             groupsDict[medoidsList[i].index[0]] = nearestClustersGlobal[
#                 nearestClustersGlobal[0] == medoidsList[i].index[0]].index.tolist()
#         # groupsList is a 2d List of the clusters that doesn't specify the medoid
#         groupsList = []
#         for i in list(groupsDict.keys()):
#             groupsList.append(groupsDict[i])
#         print(groupsDict)
#         print(medoidsList)
#
#         # adding medoid column for plotting purposes
#         dfp['medoid'] = 0
#         for key, value in groupsDict.items():
#             dfp.loc[value, 'medoid'] = key
#
#         self.centers = []
#         self.members = []
#         for medoid, members in groupsDict.items():
#             self.centers.append(medoid)
#             self.members.append(members)
#         return self.centers, self.members, len(dfp), dfp, medoidsList
#
# if __name__ == '__main__':
#     parallelMedoids = ParallelMedoids2()
#     parallelMedoids.fit()
