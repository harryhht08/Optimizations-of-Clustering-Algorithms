import pytest

from gam_package.gam import GAM

class TestClass:
    """
    def test_gam_kmedoids_mice_runtime(self):
        local_attribution_path = '../data/mice_protein.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method=None)
        g.generate()
        print("Original Medoids Algorithm Runtime: ", g.duration)

    def test_gam_parallel_mice_runtime(self):
        local_attribution_path = '../data/mice_protein.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="parallel medoids")
        g.generate()
        print("Parallel Medoids Runtime: ", g.duration)

    def test_gam_ranked_mice_runtime(self):
        local_attribution_path = '../data/mice_protein.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="ranked medoids")
        g.generate()
        print("Ranked Medoids Runtime: ", g.duration)
    """
    def test_gam_bandit_mice_runtime_200_samples(self):
        local_attribution_path = '../data/mice_protein.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="bandit pam", num_samp=200, dataset='mice_protein')
        g.generate()
        print("BanditPAM 200 Runtime: ", g.duration)

    def test_gam_bandit_mice_runtime_700_samples(self):
        local_attribution_path = '../data/mice_protein.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="bandit pam", num_samp=700, dataset='mice_protein')
        g.generate()
        print("BanditPAM 700 Runtime: ", g.duration)

    def test_gam_kernel_mushrooms_runtime(self):
        local_attribution_path = '../data/mice_protein.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="kernel medoids", dataset='mice_protein')
        g.generate()
        print("Kernel Runtime: ", g.duration)


    def test_gam_parallel_mice_bestk(self):
        local_attribution_path = '../data/mice_protein.csv'
        bestClusterNumber = 0
        bestScore = -2
        for k in range(2,5):
            g = GAM(attributions_path=local_attribution_path, n_clusters=k, cluster_method="parallel medoids", dataset= 'mice_protein')
            g.generate()
            if g.avg_silhouette_score > bestScore:
                bestScore = g.avg_silhouette_score
                bestClusterNumber = k
        print("Best Number of Clusters: ", bestClusterNumber)
        print("Best Silhouette Score:, ", bestScore)
