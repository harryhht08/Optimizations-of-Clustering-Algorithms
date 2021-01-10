"""
ranked_medoids contains the class RankedMedoids, which is used to run the ranked algorithm based on this paper:
    Ranked k-medoids: A fast and accurate rank-based partitioning algorithm for clustering large datasets

RankedMedoids can be initialized with desired arguments from GAM

The fit function is called from GAM

The fit function calls cluster, which calls main

main is the method that holds the ranked algorithm

The algorithm creates a rank and similarity matrix once at the beginning, and then uses those tables for all future
iterations in the update step, minimizing calculations needed
"""


import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pandas.plotting import parallel_coordinates
from timeit import default_timer

class RankedMedoids:
    def __init__(self, n_clusters=1, dist_func='euclidean', max_iter=1000, tol=0.0001,
                 attributions_path="mushroom-attributions-200-samples.csv"):  # conf = SparkConf().setAppName("project-gam")
        self.filepath = attributions_path
        self.k = n_clusters

        self.max_iter = max_iter
        self.tol = tol
        self.centers = None
        self.members = None

    # Attain the original dataset for the analysis
    def getDataset(self):
        data = pd.read_csv(self.filepath)
        dataset = []
        l = len(data)
        for i in range(l):
            dataset.append(data.loc[i])
        return dataset

    # Compute distance between each pair of data points
    # data: the actual data we use for this algorithm, it is a collection of data points, each point has dimension dims
    def computeDistance(self, i, j, data):
        dims = len(data[0])
        dis = 0
        for x in range(dims):
            dis += (data[i][x] - data[j][x]) ** 2
        return dis ** .5

    # Build Rank and Similarity tables. This function produces the crucial tables this algorithm requires
    def buildRankTable(self, data):
        n = len(data)
        rank = [[(0, 0) for i in range(n)] for j in range(n)]
        similarityMetrix = [[(0, 0) for i in range(n)] for j in range(n)]

        # Compute the distance and record them in a tuple
        for i in range(n):
            for j in range(i, n):
                d = self.computeDistance(i, j, data)
                rank[i][j] = (d, j)
                rank[j][i] = (d, i)

        for i in range(n):
            rank[i].sort(key=lambda x: x[0])  # Sort based on distances
            r = 0
            rankings = []
            for j in range(n):
                rankings.append(r)
                if j <= n - 2 and rank[i][j][0] == rank[i][j + 1][0]:
                    r -= 1
                r += 1
                similarityMetrix[i][j] = rank[i][j][1]  # Store the sample points
                rank[i][j] = rank[i][j][1]

            newList = [0] * n
            for j in range(n):
                newList[rank[i][j]] = rankings[j]

            rank[i] = newList

        return rank, similarityMetrix

    # Compute the hostility value of a data point
    def getHv(self, curr, m, rankTable, inside, outside):
        hv = m * (m + 1.0) / 2
        rankList = rankTable[curr]
        for ins in inside:
            if ins != curr:  # Need to exempt pairing with itself
                for outs in outside:
                    if rankList[outs] < rankList[ins]:
                        hv += 1
        return hv

    # Randomly select k data points as starter k medoids.
    def randomMedoids(self, k, data):
        medoids = set()
        for i in range(k):
            oldLen = len(medoids)
            while len(medoids) == oldLen:
                medoids.add(random.randint(0, len(data) - 1))
        return medoids

    # Assign each points to closest medoid, using Rank table
    def assignToClusters(self, k, n, medoids, rankTable):
        groups = []
        for i in range(k):
            groups.append([])  # Make k empty groups
        medoidsList = list(medoids)

        for i in range(n):
            rankRowI = rankTable[i]
            min = n
            groupIndex = n
            for j in range(k):
                m = medoidsList[j]
                if rankRowI[m] < min:
                    min = rankRowI[m]
                    groupIndex = j

            groups[groupIndex].append(i)

        return groups, medoids

    # Update medoids: use the two tables to calculate the point with highest hv value in each cluster and set it as the new medoid
    def updateMedoids(self, m, n, medoids, similarityTable, rankTable):
        newMedoids = set()
        for med in medoids:
            candidates = []
            insidePoints = similarityTable[med][:m]
            outsidePoints = similarityTable[med][m:]
            maxHv = (-1, -1)
            for ins in insidePoints:
                hv = self.getHv(ins, m, rankTable, set(insidePoints), set(outsidePoints))
                newHvPair = (ins, hv)
                if hv > maxHv[1]:
                    maxHv = (ins, hv)
                    candidates.insert(0, (ins, hv))
                else:
                    insertion = 0
                    for i in range(len(candidates)):
                        if hv > candidates[0][1]:
                            candidates.insert(insertion, newHvPair)
                            break
                        else:
                            insertion += 1

            if not maxHv[0] in newMedoids:
                newMedoids.add(maxHv[0])
            else:
                pointer = 1
                while candidates[pointer][0] in newMedoids:
                    pointer += 1
                newMedoids.add(candidates[pointer][0])

        print(newMedoids)
        return newMedoids

    def fit(self, X=None, plotit=False, verbose=True, n_clusters = 3):
        """
        Fits kmedoids with the option for plotting
        """

        start = default_timer()
        medoids, clusters, n = self.cluster(n_clusters)
        duration = default_timer() - start

        self.centers = medoids
        self.members = clusters
        if plotit:
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
        return n, duration

    # Print out the table nicely
    def printNice(self, clusters):
        for i in range(len(clusters)):
            print(clusters[i])

    # Cluster the data
    def cluster(self, n_clusters):
        dataset = self.getDataset()
        medoids, clusters = self.main(dataset, n_clusters)
        return medoids, clusters, len(dataset)

    # This method checks whether there exists same data points in different clusters
    def checkClustersContainDifferent(self, clusters):
        for i in range(len(clusters)):
            for d in clusters[i]:
                for j in range(len(clusters)):
                    if i != j:
                        if d in clusters[j]:
                            return False
        return True

    # Perform PCA and plot
    def pcaDataAndPlot(self, x, clusters):
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
        colors = ['blue', 'black', 'yellow', 'green', 'purple']
        plt.figure(figsize=(10, 8))
        plt.xlim(-3, 3)
        for i in clusters[0]:
            plt.scatter(principalDf.loc[i][0], principalDf.loc[i][1], c=colors[0])
        for i in clusters[1]:
            plt.scatter(principalDf.loc[i][0], principalDf.loc[i][1], c=colors[1])
        for i in clusters[2]:
            plt.scatter(principalDf.loc[i][0], principalDf.loc[i][1], c=colors[2])
        for i in clusters[3]:
            plt.scatter(principalDf.loc[i][0], principalDf.loc[i][1], c=colors[3])
        for i in clusters[4]:
            plt.scatter(principalDf.loc[i][0], principalDf.loc[i][1], c=colors[4])

    # Do the main job of clustering. Define values for k, m, numOfLoops. When looping, update medoids accordingly
    def main(self, data, n_clusters):
        n = len(data)
        numOfLoops = 100
        k = n_clusters
        m = 15
        medoids = self.randomMedoids(k, data)
        rankTable, similarityMetrix = self.buildRankTable(data)
        medoidsAttained = set()
        for i in range(numOfLoops):
            medoids = self.updateMedoids(m, n, medoids, similarityMetrix, rankTable)
            if medoids in medoidsAttained:
                break
            else:
                medoidsAttained.add(frozenset(medoids))
        clusters, medoids = self.assignToClusters(k, n, medoids, rankTable)

        self.printNice(clusters)
        # print(self.checkClustersContainDifferent(clusters))
        # self.plotParralelAndRadar(clusters, data)
        return medoids, clusters


if __name__ == '__main__':
    # data = getDataset()
    # main(data)

    # test2D()

    rankedMedoids = RankedMedoids()
    rankedMedoids.cluster()



