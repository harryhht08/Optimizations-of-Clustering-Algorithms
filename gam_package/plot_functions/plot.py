"""
Contains severl methods designed for plotting purposes

ldaClusterPlot - dimension reduction plot

parallelPlot - parallel plot of all points

radarPlot - singular radar plot with explanations for all medoids

facetedRadarPlot - seperate radar plot for each medoid

silhouetteAnalysis - performs silhouette analysis and gives a silhouette plot

"""

from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from math import pi
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import numpy as np
from spectral_clustering.core.util import get_scale, spectral_clustering
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def ldaClusterPlot(clusters, subpopulations, clustering_attributions):
    r = lambda: random.randint(0, 255)
    y = np.array(subpopulations)
    X = np.array(clustering_attributions)
    X_norm = (X - X.min()) / (X.max() - X.min())
    lda = LDA(n_components=2)  # 2-dimensional LDA
    lda_transformed = pd.DataFrame(lda.fit_transform(X_norm, y))

    lenClusters = len(clusters.centers)
    for medoidIndex in range(lenClusters):
        classLabel = 'Class ' + str(medoidIndex)
        randomColorHexString = '#%02X%02X%02X' % (r(), r(), r())
        plt.scatter(lda_transformed[y == medoidIndex][0],
                    lda_transformed[y == medoidIndex][1],
                    label=classLabel, c=randomColorHexString)
    # Display legend and show plot
    plt.legend(loc=3)
    plt.show()


######################################################
def parallelPlot(dfp):
    parallel_coordinates(dfp, class_column='medoid', colormap=get_cmap("Set1"))
    plt.show()

def radarPlot(dfp, medoidsList, dataset):
    if dataset == 'data/mushrooms.csv':
        # create background
        col = dfp.pop('medoid')
        dfp.insert(0, 'medoid', col)
        categories = list(dfp)[1:]
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories)
        ax.set_rlabel_position(0)
        plt.yticks([.1, .2, .3], [".1", ".2", ".3"], color="grey", size=7)
        plt.ylim(0, .3)
        # add plots
        for i in range(len(medoidsList)):
            medoidname = medoidsList[i].index[0]
            values = medoidsList[i].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=medoidname)
            ax.fill(angles, values, 'b', alpha=0.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.show()
    elif dataset == 'data/crime_without_states.csv' or dataset == '../data/crime_without_states.csv':
        col = dfp.pop('medoid')
        dfp.insert(0, 'medoid', col)
        categories = list(dfp)[1:]
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories)
        ax.set_rlabel_position(0)
        plt.yticks([.1, .2, .3, .4, .5, .6, .7 , .8 , .9, 1], [".1", ".2", ".3", ".4", ".5", ".6", ".7" , ".8" , ".9", "1"], color="grey", size=7)
        plt.ylim(0, 1)
        # add plots
        for i in range(len(medoidsList)):
            medoidname = medoidsList[i].index[0]
            values = medoidsList[i].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=medoidname)
            ax.fill(angles, values, 'b', alpha=0.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.show()

def facetedRadarPlot(dfp, medoidsList, dataset):
    if dataset == 'data/mushrooms.csv':
        # creates a radar plot for the given medoid
        def make_spider(row, title, color):
            col = dfp.pop('medoid')
            dfp.insert(0, 'medoid', col)
            categories = list(dfp)[1:]
            N = len(categories)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            ax = plt.subplot(2, 2, row + 1, polar=True, )
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            plt.xticks(angles[:-1], categories, color='grey', size=8)
            ax.set_rlabel_position(0)
            plt.yticks([.1, .2, .3], [".1", ".2", ".3"], color="grey", size=7)
            plt.ylim(0, .3)
            medoidname = medoidsList[row].index[0]
            values = medoidsList[row].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=medoidname)
            ax.fill(angles, values, 'b', alpha=0.1)
            plt.title(title, size=11, color=color, y=1.1)

        my_dpi = 96
        plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
        my_palette = get_cmap("Set2", len(medoidsList))

        #for each medoid call make_spider to create a radar plot
        for row in range(len(medoidsList)):
            make_spider(row=row, title=medoidsList[row].index[0], color=my_palette(row))
        plt.show()
    elif dataset == 'data/crime_without_states.csv' or dataset == '../data/crime_without_states.csv':
        # creates a radar plot for the given medoid
        def make_spider(row, title, color):
            col = dfp.pop('medoid')
            dfp.insert(0, 'medoid', col)
            categories = list(dfp)[1:]
            N = len(categories)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            ax = plt.subplot(2, 2, row + 1, polar=True, )
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            plt.xticks(angles[:-1], categories, color='grey', size=8)
            ax.set_rlabel_position(0)
            plt.yticks([.1, .2, .3, .4, .5, .6, .7 , .8 , .9, 1], [".1", ".2", ".3", ".4", ".5", ".6", ".7" , ".8" , ".9", "1"], color="grey", size=7)
            plt.ylim(0, 1)
            medoidname = medoidsList[row].index[0]
            values = medoidsList[row].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=medoidname)
            ax.fill(angles, values, 'b', alpha=0.1)
            plt.title(title, size=11, color=color, y=1.1)

        my_dpi = 96
        plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
        my_palette = get_cmap("Set2", len(medoidsList))

        # for each medoid call make_spider to create a radar plot
        for row in range(len(medoidsList)):
            make_spider(row=row, title=medoidsList[row].index[0], color=my_palette(row))
        plt.show()

def silhouetteAnalysis(dfp, k, centers):
    labels = []
    for i in range(len(dfp)):
        labels.append(int(dfp.iloc[i]['medoid']))
    npLabels = np.asarray(labels, dtype = np.int32)
    dfpNoMedoids = dfp.drop('medoid', 1).values
    avgSilhouetteScore = silhouette_score(dfpNoMedoids, npLabels)
    print("Average Silhouette Score: ", avgSilhouetteScore)
    sample_silhouette_values = silhouette_samples(dfpNoMedoids, npLabels)

    fig, ax = plt.subplots()
    fig.set_size_inches(18, 7)
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(dfpNoMedoids) + (k + 1) * 10])
    y_lower = 10
    color_num = 0
    for i in centers:
        color_num = color_num + 1
        ith_cluster_silhouette_values = sample_silhouette_values[npLabels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(color_num) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=avgSilhouetteScore, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()

    return avgSilhouetteScore
