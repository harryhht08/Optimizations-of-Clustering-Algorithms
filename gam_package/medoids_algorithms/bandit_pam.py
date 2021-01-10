"""
bandit_pam contains the class BanditPAM which is used to run the banditPAM algorithm based off of the paper:
    Bandit-PAM: Almost Linear Time k-Medoids Clustering via Multi-Armed Bandits

banditPAM can be initialized by GAM with the desired arguments

The fit function is called from gam

The fit function calls makeClusters, which sets the arguments and then runs build_and_swap

build_and_swap is the main function that holds the banditPAM algorithm

bandit-PAM runs in almost linear time and the time is dependent on the number of samples used in the algorithm, which
should be less than the total number of samples

"""

import csv
import os
import sys
import numpy as np
import mnist
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import pickle

from zss import simple_distance, Node
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances
import itertools
import shlex
from timeit import default_timer
from gam_package import preprocessor
from gam_package.preprocessor.preprocessor import setArguments, load_data


class BanditPAM:

    def __init__(self, n_clusters=1, max_iter=1000, tol=0.0001,
                 attributions_path="data/mushrooms.csv"):
        self.attributions_path = attributions_path
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centers = None
        self.members = None
        self.DECIMAL_DIGITS = 5
        self.SIGMA_DIVISOR = 1

    def init_logstring(self):
        print("data_utils -> init_logstring")
        '''
        Create an empty logstring with the desired fields. The logstrings will be
        updated by the algorithms.
        '''

        logstring = {
            'loss': {},
            'compute_exactly': {},
            'p': {},
            'sigma': {},
            'swap': {},
        }
        return logstring

    def update_logstring(self, logstring, k, best_distances, compute_exactly, p, sigma, swap=None):
        print("data_utils -> update_logstring")
        '''
        Update a given logstring (python dict) with the results of a BUILD or SWAP
        iteration.
        '''

        logstring['loss'][k] = np.mean(best_distances)
        logstring['compute_exactly'][k] = compute_exactly
        logstring['p'][k] = p

        if type(sigma) == list:
            logstring['sigma'][k] = ""
            logstring['sigma'][k] += " min: " + str(round(sigma[0], self.DECIMAL_DIGITS))
            logstring['sigma'][k] += " 25th: " + str(round(sigma[1], self.DECIMAL_DIGITS))
            logstring['sigma'][k] += " median: " + str(round(sigma[2], self.DECIMAL_DIGITS))
            logstring['sigma'][k] += " 75th: " + str(round(sigma[3], self.DECIMAL_DIGITS))
            logstring['sigma'][k] += " max: " + str(round(sigma[4], self.DECIMAL_DIGITS))
            logstring['sigma'][k] += " mean: " + str(round(sigma[5], self.DECIMAL_DIGITS))
        else:
            logstring['sigma'][k] = sigma

        if swap is not None:
            logstring['swap'][k] = str(swap[0]) + ',' + str(swap[1])

        return logstring

    def empty_counter(self):
        # print("data_utils -> empty_counter")
        '''
        Empty function that is called once for every distance call. Allows for easy
        counting of the number of distance calls.
        '''
        pass

    def d(self, x1, x2, metric=None):
        # print("data_utils -> d")
        '''
        Computes the distance between x1 and x2. If x2 is a list, computes the
        distance between x1 and every x2.
        '''
        assert len(x1.shape) == len(x2.shape), "Arrays must be of the same dimensions in distance computation"
        if len(x1.shape) > 1:
            assert x1.shape[0] == 1, "X1 is misshapen!"
            for _unused1 in range(x1.shape[0]):
                for _unused2 in range(x2.shape[0]):
                    self.empty_counter()  # Counts the number of distance computations

            if metric == "L2":
                return np.linalg.norm(x1 - x2, ord=2, axis=1)
            elif metric == "L1":
                return np.linalg.norm(x1 - x2, ord=1, axis=1)
            elif metric == "COSINE":
                return pairwise_distances(x1, x2, metric='cosine').reshape(-1)
            else:
                raise Exception("Bad metric specified")

        else:
            assert x1.shape == x2.shape
            assert len(x1.shape) == 1
            self.empty_counter()  # Counts the number of distance computations

            if metric == "L2":
                return np.linalg.norm(x1 - x2, ord=2)
            elif metric == "L1":
                return np.linalg.norm(x1 - x2, ord=1)
            elif metric == "COSINE":
                return cosine(x1, x2)
            else:
                raise Exception("Bad metric specified")

    def d_tree(self, x1, x2, metric=None, dist_mat=None):
        print("data_utils -> d_tree")
        '''
        Use this function for computing the edit distance between two trees.
        Supports both on-the-fly computation (metric == 'TREE') as well as using the
        precomputed distance matrix (metric == 'PRECOMP')
        '''

        if metric == 'TREE':
            # Compute the tree edit distance on-the-fly
            assert metric == 'TREE', "Bad args to tree distance fn"
            assert type(x1) == Node, "First arg must always be a single node"
            if type(x2) == Node:
                self.empty_counter()  # 1-on-1 comparison
                return simple_distance(x1, x2)
            elif type(x2) == np.ndarray:
                for _unused in x2:
                    self.empty_counter()
                return np.array([simple_distance(x1, x2_elem) for x2_elem in x2])
            else:
                raise Exception("Bad x2 type tree distance fn")
        elif metric == 'PRECOMP':
            # Use the precomputed distance matrix
            assert dist_mat is not None, "Must pass distance matrix!"
            assert type(x1) == int or type(x1) == np.int64, "Must pass x1 as an int"
            if type(x2) == int or type(x2) == np.int64:
                self.empty_counter()  # 1-on-1 comparison
                return dist_mat[x1, x2]
            elif type(x2) == np.ndarray:
                for _unused in x2:
                    self.empty_counter()
                return np.array([dist_mat[x1, x2_elem] for x2_elem in x2])
            else:
                raise Exception("Bad x2 type tree distance fn", type(x2))
        else:
            raise Exception('Bad metric argument to tree distance function')

    def cost_fn(self, dataset, tar_idx, ref_idx, best_distances, metric=None, use_diff=True, dist_mat=None):
        # print("data_utils -> cost_fn")
        '''
        Returns the "cost" of adding the pointpoint tar as a medoid:
        distances from tar to ref if it's less than the existing best distance,
        best_distances[ref_idx] otherwise

        This is called by the BUILD step of naive PAM and BanditPAM (ucb_pam).

        Contains special cases for handling trees, both with precomputed distance
        matrix and on-the-fly computation.
        '''
        if metric == 'TREE':
            assert type(dataset[tar_idx]) == Node, "Misshapen!"
            if use_diff:
                return np.minimum(self.d_tree(dataset[tar_idx], dataset[ref_idx], metric), best_distances[ref_idx]) - \
                       best_distances[ref_idx]
            return np.minimum(self.d_tree(dataset[tar_idx], dataset[ref_idx], metric), best_distances[ref_idx])
        elif metric == 'PRECOMP':
            assert type(dataset[tar_idx]) == Node, "Misshapen!"
            # Need to pass indices of nodes instead of nodes themselves
            if use_diff:
                return np.minimum(self.d_tree(tar_idx, ref_idx, metric, dist_mat), best_distances[ref_idx]) - best_distances[
                    ref_idx]
            return np.minimum(self.d_tree(tar_idx, ref_idx, metric, dist_mat), best_distances[ref_idx])
        else:
            if use_diff:
                return np.minimum(self.d(dataset[tar_idx].reshape(1, -1), dataset[ref_idx], metric),
                                  best_distances[ref_idx]) - best_distances[ref_idx]
            return np.minimum(self.d(dataset[tar_idx].reshape(1, -1), dataset[ref_idx], metric), best_distances[ref_idx])

    def cost_fn_difference(self, imgs, swaps, tmp_refs, current_medoids, metric=None):
        print("data_utils -> cost_fn_difference")
        '''
        Do not use this function. Always run experiments with the FastPAM1
        optimization, because it yields the same result.

        Returns the difference in costs for the tmp_refs if we were to perform the
        swap in swaps. Let c1 = swap[0], c2 = swap[1]. Then there are 4 cases:
          - The current best distance uses c1, a currently assigned medoid, and c2 would become the new closest medoid
          - The current best distance uses c1, but swapping it to c2 would mean a totally different medoid c3 becomes the closest
          - The current best distance does NOT use c1, and c2 would become the new closest medoid
          - The current distance does NOT use c1, and c2 would also NOT be the new closest medoid, so the point is unaffected
        '''

        raise Exception('This function is no longer supported. Please use FP1')

        num_targets = len(swaps)
        reference_best_distances, reference_closest_medoids, reference_second_best_distances = get_best_distances(
            current_medoids, imgs, subset=tmp_refs, return_second_best=True, metric=metric, dist_mat=dist_mat)
        new_losses = np.zeros(num_targets)
        N = len(imgs)

        for s_idx, s in enumerate(swaps):
            raise Exception(
                "This fn does not support tree edit distance / precomp yet. May not be an issue;comment this line out if you're OK with that.")
            # WARNING: When referring to best_distances, use indices. Otherwise, use tmp_refs[indices]
            # This is because best_distance is computed above and only returns the re-indexed subset
            old_medoid = current_medoids[s[0]]
            new_medoid = s[1]
            case1 = np.where(reference_closest_medoids == old_medoid)[0]  # INDICES
            case2 = np.where(reference_closest_medoids != old_medoid)[0]  # INDICES
            # NOTE: Many redundant computations of d here -- imgs[new_medoid] is the new medoid in lots of swaps!
            new_medoid_distances = d(imgs[new_medoid].reshape(1, -1), imgs[tmp_refs], metric)
            new_losses[s_idx] += np.sum(
                np.minimum(new_medoid_distances[case1], reference_second_best_distances[case1]))  # case1
            new_losses[s_idx] += np.sum(
                np.minimum(new_medoid_distances[case2], reference_best_distances[case2]))  # case2

        new_losses /= len(tmp_refs)

        return new_losses

    def cost_fn_difference_FP1(self, imgs, swaps, tmp_refs, current_medoids, metric=None, return_sigma=False, use_diff=True,
                               dist_mat=None):
        print("data_utils -> cost_fn_difference_FP1")
        '''
        Returns the new losses if we were to perform the swaps in swaps, as in
        cost_fn_difference above, but using the FastPAM1 optimization.

        NOTE:
        The FastPAM1 optimization consists of two mini-optimizations:
            (a) Cache d(x_old, x_ref) for every pair x_old and x_ref, since this doesn't change with x_n -- and keep track of the second best distance in case you're gonna use that
            (b) Cache d(x_new, x_ref) for every pair x_new and x_ref, since this doesn't change with old
        Then compute Delta_TD for every pair (x_old, x_new) using these CACHED values

        Both (a) and (b) are implemented.

        See cases in comment for cost_fn_difference; same cases appear here.
        '''
        num_targets = len(swaps)
        reference_best_distances, reference_closest_medoids, reference_second_best_distances = self.get_best_distances(
            current_medoids, imgs, subset=tmp_refs, return_second_best=True, metric=metric, dist_mat=dist_mat)

        new_losses = np.zeros(num_targets)
        sigmas = np.zeros(num_targets)

        N = len(imgs)

        distinct_new_medoids = set([s[1] for s in swaps])
        ALL_new_med_distances = np.zeros(
            (len(distinct_new_medoids), len(tmp_refs)))  # WARNING: Re-indexing distinct elems!!
        reidx_lookup = {}
        for d_n_idx, d_n in enumerate(distinct_new_medoids):
            reidx_lookup[d_n] = d_n_idx  # Smarter way to do this?
            if metric == 'TREE':
                ALL_new_med_distances[d_n_idx] = self.d_tree(imgs[d_n], imgs[tmp_refs], metric)
            elif metric == 'PRECOMP':
                # Must pass indices to precomp instead of nodes
                ALL_new_med_distances[d_n_idx] = self.d_tree(d_n, tmp_refs, metric, dist_mat)
            else:
                ALL_new_med_distances[d_n_idx] = self.d(imgs[d_n].reshape(1, -1), imgs[tmp_refs], metric)

        for s_idx, s in enumerate(swaps):
            # WARNING: When referring to best_distances, use indices. Otherwise, use tmp_refs[indices]
            # This is because best_distance is computed above and only returns the re-indexed subset
            old_medoid = current_medoids[s[0]]
            new_medoid = s[1]
            case1 = np.where(reference_closest_medoids == old_medoid)[0]  # List of indices
            case2 = np.where(reference_closest_medoids != old_medoid)[0]  # List of indices
            new_medoid_distances = ALL_new_med_distances[reidx_lookup[new_medoid]]
            case1_losses = np.minimum(new_medoid_distances[case1], reference_second_best_distances[case1])
            case2_losses = np.minimum(new_medoid_distances[case2], reference_best_distances[case2])

            if use_diff:
                case1_losses -= reference_best_distances[case1]
                case2_losses -= reference_best_distances[case2]

            new_losses[s_idx] = np.sum(case1_losses) + np.sum(case2_losses)

            if return_sigma:
                sigmas[s_idx] = np.std(np.hstack((case1_losses, case2_losses))) / self.SIGMA_DIVISOR

        new_losses /= len(tmp_refs)

        if return_sigma:
            return new_losses, sigmas

        return new_losses

    def get_best_distances(self, medoids, dataset, subset=None, return_second_best=False, metric=None, dist_mat=None):
        print("data_utils -> get_best_distances")
        '''
        For each point, calculate the minimum distance to any medoid.

        Do not call this from random fns which subsample the dataset, or your
        indices will be thrown off.
        '''
        assert len(medoids) >= 1, "Need to pass at least one medoid"
        assert not (return_second_best and len(
            medoids) < 2), "Need at least 2 medoids to avoid infs when asking for return_second_best"

        if metric == 'TREE':
            inner_d_fn = self.d_tree
        elif metric == 'PRECOMP':
            inner_d_fn = self.d_tree
            assert dist_mat is not None, "Need to pass dist_mat to get_best_distances"
        else:
            inner_d_fn = self.d

        if subset is None:
            N = len(dataset)
            refs = range(N)
        else:
            refs = subset

        # NOTE: Use a Heap or sorted linked list for best distance, second best
        # distance, third best distance, etc and pop as necessary if doing multiple
        # swaps

        best_distances = np.array([float('inf') for _ in refs])
        second_best_distances = np.array([float('inf') for _ in refs])
        closest_medoids = np.array([-1 for _ in refs])

        # NOTE: Could speed this up with array broadcasting and taking min across medoid axis
        for p_idx, point in enumerate(refs):
            for m in medoids:
                # WARNING: If dataset has been shuffled, than the medoids will refer to the WRONG medoids!!!
                if metric == 'PRECOMP':
                    # NOTE: Can probably consolidate this with case below by just saying dist_mat = None if not precomp
                    if inner_d_fn(m, point, metric, dist_mat) < best_distances[p_idx]:
                        second_best_distances[p_idx] = best_distances[p_idx]
                        best_distances[p_idx] = inner_d_fn(m, point, metric, dist_mat)
                        closest_medoids[p_idx] = m
                    elif inner_d_fn(m, point, metric, dist_mat) < second_best_distances[p_idx]:
                        # Reach this case if the new medoid is between current 2nd and first, but not better than first
                        second_best_distances[p_idx] = inner_d_fn(m, point, metric, dist_mat)
                else:
                    if inner_d_fn(dataset[m], dataset[point], metric) < best_distances[p_idx]:
                        second_best_distances[p_idx] = best_distances[p_idx]
                        best_distances[p_idx] = inner_d_fn(dataset[m], dataset[point], metric)
                        closest_medoids[p_idx] = m
                    elif inner_d_fn(dataset[m], dataset[point], metric) < second_best_distances[p_idx]:
                        # Reach this case if the new medoid is between current 2nd and first, but not better than first
                        second_best_distances[p_idx] = inner_d_fn(dataset[m], dataset[point], metric)

        if return_second_best:
            return best_distances, closest_medoids, second_best_distances
        return best_distances, closest_medoids

    # TODO: Explicitly pass metric instead of args.metric here
    def medoid_swap(self, medoids, best_swap, imgs, loss, args, dist_mat=None):
        print("\nmedoid_swap")
        '''
        Swaps the medoid-nonmedoid pair in best_swap if it would lower the loss on
        the datapoints in imgs. Returns a string describing whether the swap was
        performed, as well as the new medoids and new loss.
        '''

        # NOTE Store these explicitly to avoid incorrect reference after medoids have been updated when printing
        orig_medoid = medoids[best_swap[0]]
        new_medoid = best_swap[1]

        new_medoids = medoids.copy()
        new_medoids.remove(orig_medoid)
        new_medoids.append(new_medoid)
        new_best_distances, new_closest_medoids \
            = self.get_best_distances(new_medoids, imgs, metric=args.metric, dist_mat=dist_mat)

        new_loss = np.mean(new_best_distances)
        performed_or_not = ''
        if new_loss < loss:
            performed_or_not = "SWAP PERFORMED"
            print("SWAP PERFORMED")
            swap_performed = True
        else:
            performed_or_not = "NO SWAP PERFORMED"
            print("NO SWAP PERFORMED")
            new_medoids = medoids

        if args.verbose >= 1:
            print("Tried to swap", orig_medoid, "with", new_medoid)
            print(performed_or_not)
            print("Old loss:", loss)
            print("New loss:", new_loss)

        print("updated medoids: ", new_medoids)
        return performed_or_not, new_medoids, min(new_loss, loss)

    def build_sample_for_targets(self, imgs, targets, batch_size, best_distances, metric=None, return_sigma=False,
                                 dist_mat=None):
        print("ucb_pam -> build_sample_for_targets")
        '''
        For the given targets, which are candidate points to be assigned as medoids
        during a build step, we compute the changes in loss they would induce
        on a subsample of batch_size reference points (tmp_refs).

        The returned value is an array of estimated changes in loss for each target.
        '''

        # TODO: Improve this with array broadcasting
        N = len(imgs)
        estimates = np.zeros(len(targets))
        sigmas = np.zeros(len(targets))
        tmp_refs = np.array(np.random.choice(N, size=batch_size, replace=False), dtype='int')
        for tar_idx, target in enumerate(targets):
            if best_distances[0] == np.inf:
                # No medoids have been assigned, can't use the difference in loss
                costs = self.cost_fn(imgs, target, tmp_refs, best_distances, metric=metric, use_diff=False,
                                dist_mat=dist_mat)
            else:
                costs = self.cost_fn(imgs, target, tmp_refs, best_distances, metric=metric, use_diff=True, dist_mat=dist_mat)

            estimates[tar_idx] = np.mean(costs)
            if return_sigma:
                sigmas[tar_idx] = np.std(costs) / self.SIGMA_DIVISOR

        if return_sigma:
            return estimates.round(self.DECIMAL_DIGITS), sigmas, tmp_refs

        return estimates.round(self.DECIMAL_DIGITS), None, tmp_refs

    def build(self, args, imgs, sigma, dist_mat=None):
        print("ucb_pam -> build")
        '''
        Performs the BUILD step of BanditPAM. Analogous to the BUILD step of PAM,
        BanditPAM assigns the initial medoids one-by-one by choosing the point at
        each step that would lower the total loss the most. Instead of computing the
        change in loss for every other point, it estimates these changes in loss.
        '''

        B_logstring = self.init_logstring()

        ### Parameters
        metric = args.metric
        N = len(imgs)
        p = 1. / (N * 1000)
        num_samples = np.zeros(N)
        estimates = np.zeros(N)

        cache_computed = np.zeros((N, N))

        if len(args.warm_start_medoids) > 0:
            warm_start_medoids = list(map(int, args.warm_start_medoids.split(',')))
            medoids = warm_start_medoids.copy()
            num_medoids_found = len(medoids)
            best_distances, closest_medoids = np.array(
                self.get_best_distances(medoids, imgs, metric=metric, dist_mat=dist_mat))
        else:
            medoids = []
            num_medoids_found = 0
            best_distances = np.inf * np.ones(N)

        for k in range(num_medoids_found, args.num_medoids):
            compute_sigma = True

            if args.verbose >= 1:
                print("Finding medoid", k)

            ## Initialization
            step_count = 0
            candidates = range(N)  # Initially, consider all points
            lcbs = 1000 * np.ones(N)
            ucbs = 1000 * np.ones(N)
            T_samples = np.zeros(N)
            exact_mask = np.zeros(N)
            sigmas = np.zeros(N)

            original_batch_size = 100
            base = 1  # Right now, use constant batch size

            while (len(candidates) > 0):
                if args.verbose >= 1:
                    print("Step count:", step_count, ", Candidates:", len(candidates), candidates)

                this_batch_size = int(original_batch_size * (base ** step_count))

                # Find the points whose change in loss should be computed exactly,
                # because >= N reference points have already been sampled.
                compute_exactly = np.where((T_samples + this_batch_size >= N) & (exact_mask == 0))[0]
                if len(compute_exactly) > 0:
                    if args.verbose >= 1:
                        print("COMPUTING EXACTLY ON STEP COUNT", step_count)

                    estimates[compute_exactly], _, calc_refs = self.build_sample_for_targets(imgs, compute_exactly, N,
                                                                                        best_distances, metric=metric,
                                                                                        return_sigma=False,
                                                                                        dist_mat=dist_mat)
                    lcbs[compute_exactly] = estimates[compute_exactly]
                    ucbs[compute_exactly] = estimates[compute_exactly]
                    exact_mask[compute_exactly] = 1
                    T_samples[compute_exactly] += N
                    candidates = np.setdiff1d(candidates,
                                              compute_exactly)  # Remove compute_exactly points from candidates so they're bounds don't get updated below

                    for c_e in compute_exactly:
                        for c_r in calc_refs:
                            cache_computed[c_e, c_r] = 1

                if len(candidates) == 0: break  # The last remaining candidates were computed exactly

                # Gather more evaluations of the change in loss for some reference points
                if compute_sigma:
                    # Estimate sigma from the data, if necessary
                    sample_costs, sigmas, calc_refs = self.build_sample_for_targets(imgs, candidates, this_batch_size,
                                                                               best_distances, metric=metric,
                                                                               return_sigma=True, dist_mat=dist_mat)
                    compute_sigma = False
                else:
                    sample_costs, _, calc_refs = self.build_sample_for_targets(imgs, candidates, this_batch_size,
                                                                          best_distances, metric=metric,
                                                                          return_sigma=False, dist_mat=dist_mat)

                for c_e in candidates:
                    for c_r in calc_refs:
                        cache_computed[c_e, c_r] = 1

                if args.verbose >= 1:
                    print("Unique distances computed:", np.sum(cache_computed))

                # Update running average of estimates and confidence bounce
                estimates[candidates] = \
                    ((T_samples[candidates] * estimates[candidates]) + (this_batch_size * sample_costs)) / (
                                this_batch_size + T_samples[candidates])
                T_samples[candidates] += this_batch_size
                cb_delta = sigmas[candidates] * np.sqrt(np.log(1 / p) / T_samples[candidates])
                lcbs[candidates] = estimates[candidates] - cb_delta
                ucbs[candidates] = estimates[candidates] + cb_delta

                candidates = np.where((lcbs < ucbs.min()) & (exact_mask == 0))[0]
                step_count += 1

            # Breaks exact ties with first. Also converts array to int.
            # This does indeed happen, for example in ucb k = 50, n = 100, s = 42, d = MNIST
            new_medoid = np.arange(N)[np.where(lcbs == lcbs.min())]
            new_medoid = new_medoid[0]

            if args.verbose >= 1:
                print("New Medoid:", new_medoid)

            medoids.append(new_medoid)
            best_distances, closest_medoids = self.get_best_distances(medoids, imgs, metric=metric, dist_mat=dist_mat)
            # print("updated medoids: ", closest_medoids)
            print("Computed exactly for:", exact_mask.sum())

            # get information about sigmas: min, 25, median, 75, max, mean
            sigma_arr = [np.min(sigmas), np.quantile(sigmas, 0.25), np.median(sigmas), np.quantile(sigmas, 0.75),
                         np.max(sigmas), np.mean(sigmas)]
            B_logstring = self.update_logstring(B_logstring, k, best_distances, exact_mask.sum(), p, sigma_arr)

        return medoids, B_logstring, cache_computed

    def swap_sample_for_targets(self, imgs, targets, current_medoids, batch_size, FastPAM1=False, metric=None,
                                return_sigma=False, dist_mat=None):
        print("ucb_pam -> swap_sample_for_targets")
        '''
        For the given targets (potential swaps) during a swap step, we compute the
        changes in loss they would induce on a subsample of batch_size reference
        points (tmp_refs) when the swap is performed.

        The returned value is an array of estimated changes in loss for each target
        (swap).
        '''
        # NOTE: Improve this with array broadcasting
        # Also generalize and consolidate it with the fn of the same name in the build step
        orig_medoids = targets[0]
        new_medoids = targets[1]
        assert len(orig_medoids) == len(new_medoids), "Must pass equal number of original medoids and new medoids"
        # NOTE: Need to preserve order of swaps that are passed - otherwise estimates will be for the wrong swaps
        # I.e. there will be an error if estimates aren't indexed properly -- only ok if we do 1 target at a time

        # WARNING: Zip doesn't throw an error for unequal lengths, it just drops extraneous points
        swaps = list(zip(orig_medoids, new_medoids))

        N = len(imgs)
        k = len(current_medoids)

        tmp_refs = np.array(np.random.choice(N, size=batch_size, replace=False), dtype='int')
        if FastPAM1:
            if return_sigma:
                estimates, sigmas = self.cost_fn_difference_FP1(imgs, swaps, tmp_refs, current_medoids, metric=metric,
                                                           return_sigma=True,
                                                           dist_mat=dist_mat)  # NOTE: depends on other medoids too!
                return estimates.round(self.DECIMAL_DIGITS), sigmas, tmp_refs
            else:
                estimates = self.cost_fn_difference_FP1(imgs, swaps, tmp_refs, current_medoids, metric=metric,
                                                   return_sigma=False,
                                                   dist_mat=dist_mat)  # NOTE: depends on other medoids too!
        else:
            # NOTE: Return_sigma currently only supported with FP1 trick; need to add it to cost_fn_difference too
            raise Exception("Do not use this! Doesn't support dist_mat")
            estimates = self.cost_fn_difference(imgs, swaps, tmp_refs, current_medoids, metric=metric)

        return estimates.round(self.DECIMAL_DIGITS), None, tmp_refs

    def swap(self, args, imgs, sigma, init_medoids, dist_mat=None, cache_computed=None):
        print("swap")
        '''
        Performs the SWAP step of BanditPAM. Analogous to the SWAP step of PAM,
        BanditPAM chooses medoids to swap with non-medoids by performing the swap
        that would lower the total loss the most at each step. Instead of computing
        the exact change in loss for every other point, it estimates these changes.
        '''

        S_logstring = self.init_logstring()
        metric = args.metric
        k = len(init_medoids)
        N = len(imgs)
        p = 1. / (N * k * 1000)
        max_iter = 1e4

        medoids = init_medoids.copy()
        # NOTE: best_distances is NOT updated in future rounds - the analogy from build is broken. Maybe rename the variable
        best_distances, closest_medoids = self.get_best_distances(medoids, imgs, metric=metric, dist_mat=dist_mat)
        loss = np.mean(best_distances)
        iter = 0
        swap_performed = True
        while swap_performed and iter < max_iter:  # not converged
            compute_sigma = True
            iter += 1

            candidates = np.array(list(itertools.product(range(k), range(N))))  # A candidate is a PAIR
            lcbs = 1000 * np.ones((k,
                                   N))  # NOTE: Instantiating these as np.inf gives runtime errors and nans. Find a better way to do this instead of using 1000
            estimates = 1000 * np.ones((k, N))
            ucbs = 1000 * np.ones((k, N))

            T_samples = np.zeros((k, N))
            exact_mask = np.zeros((k, N))

            original_batch_size = 100
            base = 1  # Right now, use constant batch size

            step_count = 0
            while (len(candidates) > 0):
                if args.verbose >= 1:
                    print("SWAP Step count:", step_count)

                this_batch_size = int(original_batch_size * (base ** step_count))

                # Find swaps whose returns should be computed exactly, because >= N
                # reference points have already been sampled
                comp_exactly_condition = np.where((T_samples + this_batch_size >= N) & (exact_mask == 0))
                compute_exactly = np.array(list(zip(comp_exactly_condition[0], comp_exactly_condition[1])))
                if len(compute_exactly) > 0:
                    if args.verbose >= 1:
                        print("COMPUTING EXACTLY ON STEP COUNT", step_count)

                    exact_accesses = (compute_exactly[:, 0], compute_exactly[:, 1])
                    estimates[exact_accesses], _, calc_refs = self.swap_sample_for_targets(imgs, exact_accesses, medoids, N,
                                                                                      args.fast_pam1, metric=metric,
                                                                                      return_sigma=False,
                                                                                      dist_mat=dist_mat)
                    lcbs[exact_accesses] = estimates[exact_accesses]
                    ucbs[exact_accesses] = estimates[exact_accesses]
                    exact_mask[exact_accesses] = 1
                    T_samples[exact_accesses] += N

                    cand_condition = np.where((lcbs < ucbs.min()) & (exact_mask == 0))
                    candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))

                    for e_a in zip(exact_accesses[0], exact_accesses[1]):
                        for c_r in calc_refs:
                            cache_computed[e_a[0], c_r] = 1
                            cache_computed[e_a[1], c_r] = 1

                if len(candidates) == 0: break  # The last candidates were computed exactly

                # Gather more evaluations of the change in loss for some reference points
                accesses = (candidates[:, 0], candidates[:, 1])
                if compute_sigma:
                    # Estimate sigma from the data, if necessary
                    new_samples, sigmas, calc_refs = self.swap_sample_for_targets(imgs, accesses, medoids, this_batch_size,
                                                                             args.fast_pam1, metric=metric,
                                                                             return_sigma=True, dist_mat=dist_mat)
                    sigmas = sigmas.reshape(k, N)  # So that can access it with sigmas[accesses] below
                    compute_sigma = False
                else:
                    new_samples, _, calc_refs = self.swap_sample_for_targets(imgs, accesses, medoids, this_batch_size,
                                                                        args.fast_pam1, metric=metric,
                                                                        return_sigma=False, dist_mat=dist_mat)

                for acc_ in zip(accesses[0], accesses[1]):
                    for c_r in calc_refs:
                        cache_computed[acc_[0], c_r] = 1
                        cache_computed[acc_[1], c_r] = 1

                if args.verbose >= 1:
                    print("Unique distances computed:", np.sum(cache_computed))

                # Update running average of estimates and confidence bounce
                estimates[accesses] = \
                    ((T_samples[accesses] * estimates[accesses]) + (this_batch_size * new_samples)) / (
                                this_batch_size + T_samples[accesses])
                T_samples[accesses] += this_batch_size
                # NOTE: Sigmas is contains a value for EVERY arm, even non-candidates, so need [accesses]
                cb_delta = sigmas[accesses] * np.sqrt(np.log(1 / p) / T_samples[accesses])
                lcbs[accesses] = estimates[accesses] - cb_delta
                ucbs[accesses] = estimates[accesses] + cb_delta

                cand_condition = np.where((lcbs < ucbs.min()) & (exact_mask == 0))  # BUG: Fix this since it's 2D
                candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))
                step_count += 1

            # Choose the minimum amongst all losses and perform the swap
            # NOTE: possible to get first elem of zip object without converting to list?
            best_swaps = zip(np.where(lcbs == lcbs.min())[0], np.where(lcbs == lcbs.min())[1])
            best_swaps = list(best_swaps)
            best_swap = best_swaps[0]

            old_medoid_x = medoids[best_swap[0]]
            new_medoid_x = best_swap[1]

            print("Computed exactly for:", exact_mask.sum())
            performed_or_not, medoids, loss = self.medoid_swap(medoids, best_swap, imgs, loss, args, dist_mat=dist_mat)

            if original_batch_size >= len(imgs):
                # Corner case where sigmas aren't computed for too-small datasets
                # NOTE: This is different from the build step because in the build step, the sigmas are all initialized to 0's. Should be consistent between the two.
                sigma_arr = [0, 0, 0, 0, 0, 0]
            else:
                sigma_arr = [np.min(sigmas), np.quantile(sigmas, 0.25), np.median(sigmas), np.quantile(sigmas, 0.75),
                             np.max(sigmas), np.mean(sigmas)]
            S_logstring = self.update_logstring(S_logstring, iter - 1, loss, exact_mask.sum(), p, sigma_arr,
                                           swap=(old_medoid_x, new_medoid_x))
            # TODO: Need to update swap_performed variable above, right now only breaking
            if performed_or_not == "NO SWAP PERFORMED":
                break

        return medoids, S_logstring, iter, loss

    def build_and_swap(self, args):
        print("build_and_swap")
        '''
        Run the entire BanditPAM algorithm, both the BUILD step and the SWAP step
        '''

        num_swaps = -1
        final_loss = -1
        dist_mat = None

        total_data, total_labels, sigma, feature_labels = load_data(args)
        self.feature_labels = feature_labels
        np.random.seed(args.seed)
        if args.metric == 'PRECOMP':
            dist_mat = np.loadtxt('tree-3630.dist')
            random_indices = np.random.choice(len(total_data), size=args.sample_size, replace=False)
            imgs = np.array([total_data[x] for x in random_indices])
            dist_mat = dist_mat[random_indices][:, random_indices]
        elif args.metric == 'TREE':
            imgs = np.random.choice(total_data, size=args.sample_size, replace=False)
        else:
            # Can remove range() here?
            imgs = total_data[np.random.choice(range(len(total_data)), size=args.sample_size, replace=False)]

        built_medoids = []
        B_logstring = {}
        N = len(imgs)
        if 'B' in args.build_ao_swap:
            assert args.cache_computed is None, "Cache_computed should be None"
            built_medoids, B_logstring, cache_computed = self.build(args, imgs, sigma, dist_mat=dist_mat)
            args.cache_computed = cache_computed
            print("Built medoids", built_medoids)

        swapped_medoids = []
        S_logstring = {}
        if 'S' in args.build_ao_swap:
            if built_medoids == [] and len(args.warm_start_medoids) < args.num_medoids:
                raise Exception("Invalid call to Swap step")

            if built_medoids == []:
                init_medoids = list(map(int, args.warm_start_medoids.split(',')))
                print("Swap init medoids:", init_medoids)
            else:
                init_medoids = built_medoids.copy()

            swapped_medoids, S_logstring, num_swaps, final_loss = self.swap(args, imgs, sigma,
                                                                                    init_medoids,
                                                                                    dist_mat=dist_mat,
                                                                                    cache_computed=args.cache_computed)
            print("Final medoids", swapped_medoids)

        uniq_d = np.sum(args.cache_computed)
        #return built_medoids, swapped_medoids, B_logstring, S_logstring, num_swaps, final_loss, uniq_d
        best_distances, closest_medoids = self.get_best_distances(swapped_medoids, imgs, metric=args.metric, dist_mat=dist_mat)
        groupsDict = {}
        for i in range(len(swapped_medoids)):
            groupsDict[swapped_medoids[i]] = [j for j, x in enumerate(closest_medoids) if x == swapped_medoids[i]]

        self.centers = []
        self.members = []
        for medoid, members in groupsDict.items():
            self.centers.append(medoid)
            self.members.append(members)
        return self.centers, self.members, N, imgs, self.feature_labels


    def makeClusters(self, datasetName, num_samp):
        args = setArguments(datasetName, num_samp, n_clusters = self.n_clusters)
        return self.build_and_swap(args)
    """
    def makeClusters(self, datasetName, num_samp, n_clusters):
        args = self.setArguments(datasetName, num_samp, n_clusters)
    """



    def fit(self, X = None, plotit=False, verbose=True, attributions_path = None, num_samp = 200, dataset=None):
        """
        Fits kmedoids with the option for plotting
        """
        if attributions_path is not None:
            self.attributions_path = attributions_path
        start = default_timer()
        _,_, n, imgs, feature_labels = self.makeClusters(dataset, num_samp)
        duration = default_timer() - start
        if plotit:
            _, ax = plt.subplots(1, 1)
            colors = ["b", "g", "r", "c", "m", "y", "k"]
            if self.n_clusters > len(colors):
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
        return n, imgs, feature_labels, duration



if __name__ == '__main__':
    bandit_pam = BanditPAM()
    n, imgs, feature_labels, duration = bandit_pam.fit(attributions_path="mice_protein")