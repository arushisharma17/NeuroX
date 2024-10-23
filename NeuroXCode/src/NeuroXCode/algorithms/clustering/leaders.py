import numpy as np
from annoy import AnnoyIndex
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import statistics
import time
from ...utilities.utils import save_clustering_results


class Leaders:
    """
    A clique of follower points for a leader point.
    """

    def __init__(self, p, j):
        self.members = [p]
        self.member_indices = [j]
        self.centroid = p

    def __len__(self):
        return len(self.members)

    def add(self, p, j):
        """
        Add a new follower to the clique and update the centroid.
        """
        self.centroid = (self.centroid * len(self.members) + p) / (1 + len(self.members))
        self.members.append(p)
        self.member_indices.append(j)

    def dist(self, p):
        """
        Returns the distance of point p to the centroid of the clique.
        """
        return np.linalg.norm(p - self.centroid)


class LeadersClusteringPipeline:
    def __init__(self, output_path='./output', K=5, tau=None, is_fast=True, ann_file=None):
        self.output_path = output_path
        self.K = K
        self.tau = tau
        self.is_fast = is_fast
        self.ann_file = ann_file

    @staticmethod
    def calculate_tau(t, points):
        """
        Estimate tau if not provided by calculating the median of distances.
        """
        m = np.random.choice(range(points.shape[0]), replace=False, size=1000)
        dists_tau = [t.get_nns_by_item(i, 2, include_distances=True)[1] for i in m]
        return statistics.median([d[1] for d in dists_tau])

    def fast_cliques(self, cliques, points):
        if not self.is_fast:
            if self.tau is None:
                raise ValueError("Tau must be provided when not using fast mode.")
            for j, p in enumerate(points):
                if (j - 1) % 1000 == 0:
                    print(f"Processed {j - 1} points, {np.round(len(cliques) / j * 100, 2)}% cliques formed")
                found = False
                for c in cliques:
                    if c.dist(p) < self.tau:
                        c.add(p, j)
                        found = True
                        break
                if not found:
                    cliques.append(Leaders(p, j))

    def build_or_load_annoy_index(self, points):
        """
        Builds a new Annoy index or loads an existing one from a file.
        """
        t = AnnoyIndex(points.shape[1], 'euclidean')
        if not self.ann_file:
            for i, p in enumerate(points):
                t.add_item(i, p)
            t.build(1000)
        else:
            t.load(self.ann_file)
        return t

    def estimate_tau_if_needed(self, t, points):
        """
        Estimate tau if not provided, using the pre-defined calculation method.
        """
        if self.tau is None:
            self.tau = self.calculate_tau(t, points)

    def find_neighbors(self, t, i):
        """
        Finds neighbors and dynamically adjusts the upper limit (ul) until a point is found outside the tau range.
        """
        ul = 100
        found = False
        while not found:
            neighbours, dists = t.get_nns_by_item(i, ul, include_distances=True)
            if dists[-1] < self.tau:
                ul *= 2
            else:
                for j in range(len(neighbours)):
                    if dists[j] > self.tau:
                        ul = j
                        found = True
                        break
        return neighbours[:ul]

    @staticmethod
    def add_neighbors_to_clique(cliques, points, neighbours, used_indices):
        """
        Add neighbors to the current clique and mark them as used.
        """
        cliques.append(Leaders(points[neighbours[0], :], neighbours[0]))
        used_indices[neighbours[0]] = 1
        for n in neighbours[1:]:
            if used_indices[n] == 1:
                continue
            cliques[-1].add(points[n, :], n)
            used_indices[n] = 1

    def not_fast_cliques(self, cliques, points):
        """
        Main method to compute cliques using the not-fast (Annoy-based) approach.
        """
        t = self.build_or_load_annoy_index(points)
        self.estimate_tau_if_needed(t, points)

        used_indices = [0] * points.shape[0]
        for i, p in enumerate(points):
            if used_indices[i] != 0:
                continue

            neighbours = self.find_neighbors(t, i)
            self.add_neighbors_to_clique(cliques, points, neighbours, used_indices)

            if len(cliques) % 100 == 0:
                print(f'Cliques {len(cliques)} -- Points {sum(used_indices)}/{points.shape[0]}')

    def leaders_cluster(self, points, vocab):
        """
        Runs the leaders clustering algorithm and returns the clusters.
        """
        cliques = []
        if not self.is_fast:
            self.fast_cliques(cliques, points)
        else:
            self.not_fast_cliques(cliques, points)

        centroids = [c.centroid for c in cliques]
        clustering = AgglomerativeClustering(n_clusters=self.K).fit(centroids)

        word_clusters = defaultdict(list)
        for i, label in enumerate(clustering.labels_):
            word_clusters[label].extend([vocab[u] for u in cliques[i].member_indices if u < len(vocab)])

        return clustering, word_clusters

    def run_pipeline(self, points, vocab):
        """
        Run the leaders clustering pipeline.
        """
        start_time = time.time()
        clustering, clusters = self.leaders_cluster(points, vocab)
        end_time = time.time()

        # Save clustering results
        save_clustering_results(clustering.labels_, clusters, self.output_path, self.K)

        # Log clustering process (could log to console or other logging systems in real usage)
        print(f"Clustering completed in {end_time - start_time} seconds.")
        return clustering, clusters
