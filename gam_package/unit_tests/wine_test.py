import pytest

from gam_package.gam import GAM


class TestClass:

    def test_gam_kmedoids_wine_runtime(self):
        local_attribution_path = '../data/wine_clean.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method=None)
        g.generate()
        print("Original Medoids Algorithm Runtime: ", g.duration)

    def test_gam_parallel_wine_runtime(self):
        local_attribution_path = '../data/wine_clean.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="parallel medoids")
        g.generate()
        print("Parallel Medoids Runtime: ", g.duration)

    def test_gam_ranked_wine_runtime(self):
        local_attribution_path = '../data/wine_clean.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="ranked medoids")
        g.generate()
        print("Ranked Medoids Runtime: ", g.duration)

    def test_gam_bandit_wine_runtime(self):
        local_attribution_path = '../data/wine_clean.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="bandit pam", dataset='wine')
        g.generate()
        print("BanditPAM Runtime: ", g.duration)

    def test_gam_parallel_wine_bestk(self):
        local_attribution_path = '../data/wine_clean.csv'
        bestClusterNumber = 0
        bestScore = -2
        for k in range(2,5):
            g = GAM(attributions_path=local_attribution_path, n_clusters=k, cluster_method="bandit pam", dataset='wine')
            g.generate()
            if g.avg_silhouette_score > bestScore:
                bestScore = g.avg_silhouette_score
                bestClusterNumber = k
        print("Best Number of Clusters: ", bestClusterNumber)
        print("Best Silhouette Score:, ", bestScore)
