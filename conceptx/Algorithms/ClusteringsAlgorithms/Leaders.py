import numpy as np
from annoy import AnnoyIndex
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from conceptx.Utilities.utils import load_data, save_clustering_results, log_clustering_process
import os
import statistics
import time


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
    def __init__(self, output_path='../../output', K=5, tau=None, is_fast=True, ann_file=None):
        self.output_path = output_path
        self.K = K
        self.tau = tau
        self.is_fast = is_fast
        self.ann_file = ann_file
        os.makedirs(self.output_path, exist_ok=True)

    @staticmethod
    def _calculate_tau(self, t, points):
        """
        Estimate tau if not provided by calculating the median of distances.
        """
        m = np.random.choice(range(points.shape[0]), replace=False, size=1000)
        dists_tau = [t.get_nns_by_item(i, 2, include_distances=True)[1] for i in m]
        return statistics.median([d[1] for d in dists_tau])

    def leaders_cluster(self, points, vocab):
        """
        Runs the leaders clustering algorithm and returns the clusters.
        """
        cliques = []

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
        else:
            t = AnnoyIndex(points.shape[1], 'euclidean')
            if not self.ann_file:
                for i, p in enumerate(points):
                    t.add_item(i, p)
                t.build(1000)
                t.save(f'{self.output_path}/leaders.ann')
            else:
                t.load(self.ann_file)

            # Estimate tau if not provided
            if self.tau is None:
                self.tau = self._calculate_tau(t, points)

            used_indices = [0] * points.shape[0]
            for i, p in enumerate(points):
                if used_indices[i] != 0:
                    continue
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

                cliques.append(Leaders(points[neighbours[0], :], neighbours[0]))
                used_indices[neighbours[0]] = 1
                for n in neighbours[1:ul]:
                    if used_indices[n] == 1:
                        continue
                    cliques[-1].add(points[n, :], n)
                    used_indices[n] = 1
                if len(cliques) % 100 == 0:
                    print(f'Cliques {len(cliques)} -- Points {sum(used_indices)}/{points.shape[0]}')

        centroids = [c.centroid for c in cliques]
        clustering = AgglomerativeClustering(n_clusters=self.K).fit(centroids)

        word_clusters = defaultdict(list)
        for i, label in enumerate(clustering.labels_):
            word_clusters[label].extend([vocab[u] for u in cliques[i].member_indices if u < len(vocab)])

        return clustering, word_clusters

    def save_clusters(self, clustering, clusters):
        """
        Save clustering results to output folder.
        """
        save_clustering_results(clustering, clusters, self.output_path, self.K, ref='leaders')

    def run_pipeline(self, points, vocab):
        """
        Run the leaders clustering pipeline.
        """
        start_time = time.time()
        clustering, clusters = self.leaders_cluster(points, vocab)
        end_time = time.time()

        log_clustering_process(points, vocab, self.K, start_time, end_time, clusters)
        self.save_clusters(clustering, clusters)

        return clustering, clusters


# Main function for testing
def main():
    # Load data from utils
    points, vocab = load_data(num_points=1000, num_dims=5, vocab_size=100, output_path='../../output')

    # Initialize the leaders clustering pipeline
    pipeline = LeadersClusteringPipeline(output_path='../../output', K=5, is_fast=True)

    # Run the clustering pipeline
    pipeline.run_pipeline(points, vocab)


if __name__ == "__main__":
    main()
