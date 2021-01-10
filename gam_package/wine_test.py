from gam_package.gam import GAM


class TestClass:

    # def test_gam_parallel_mushrooms_runtime(self):
    #     gam = GAM()
    #     local_attribution_path = 'data/mushroom-attributions-200-samples.csv'
    #     g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="parallel medoids")
    #     g.generate()
    #     print("Parallel Medoids Runtime: ", g.duration)

    def test_gam_ranked_mushrooms_runtime(self):
        gam = GAM()
        local_attribution_path = 'data/mushroom-attributions-200-samples.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="ranked medoids")
        g.generate()
        print("Ranked Medoids Runtime: ", g.duration)

    def test_gam_bandit_mushrooms_runtime(self):
        gam = GAM()
        local_attribution_path = 'data/mushroom-attributions-200-samples.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="bandit pam")
        g.generate()
        print("BanditPAM Runtime: ", g.duration)



