import pytest

from gam_package.gam import GAM

class TestClass:
    """
    def test_gam_kmedoids_crime_runtime(self):
        local_attribution_path = '../data/crime.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method=None)
        g.generate()
        print("Original Medoids Algorithm Runtime: ", g.duration)

    def test_gam_parallel_crime_runtime(self):
        local_attribution_path = '../data/crime.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="parallel medoids")
        g.generate()
        print("Parallel Medoids Runtime: ", g.duration)

    def test_gam_ranked_crime_runtime(self):
        local_attribution_path = '../data/crime.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="ranked medoids")
        g.generate()
        print("Ranked Medoids Runtime: ", g.duration)
    """
    def test_gam_bandit_crime_runtime_200_samples(self):
        local_attribution_path = '../data/crime.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="bandit pam", num_samp=200, dataset='crime')
        g.generate()
        print("BanditPAM 200 Runtime: ", g.duration)

    def test_gam_bandit_crime_runtime_700_samples(self):
        local_attribution_path = '../data/crime.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="bandit pam", num_samp=700, dataset='crime')
        g.generate()
        print("BanditPAM 700 Runtime: ", g.duration)

    def test_gam_bandit_crime_runtime_1000_samples(self):
        local_attribution_path = '../data/crime.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="bandit pam", num_samp=1000, dataset='crime')
        g.generate()
        print("BanditPAM 1000 Runtime: ", g.duration)

    def test_gam_parallel_crime_bestk(self):
        local_attribution_path = '../data/crime.csv'
        bestClusterNumber = 0
        bestScore = -2
        for k in range(2,5):
            g = GAM(attributions_path=local_attribution_path, n_clusters=k, cluster_method="bandit pam", dataset='crime')
            g.generate()
            if g.avg_silhouette_score > bestScore:
                bestScore = g.avg_silhouette_score
                bestClusterNumber = k
        print("Best Number of Clusters: ", bestClusterNumber)
        print("Best Silhouette Score:, ", bestScore)