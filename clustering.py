from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import util
import corpus as c

# https://www.sbert.net/examples/applications/clustering/README.html


# Cluster similar notes together
# (i.e., calculate and compare similarties between all notes)
# ...
# Agglomerative clustering.
def agglo_clustering(similarities, note_titles, nc):
    """
    nc: number of clusters to divide into.
    """
    clustering_model = AgglomerativeClustering(n_clusters=nc,
                                               distance_threshold=None)
    clustering_model.fit(similarities)
    cluster_assignment = clustering_model.labels_

    clustered_notes = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_notes:
            clustered_notes[cluster_id] = []

        clustered_notes[cluster_id].append(note_titles[sentence_id])

    for i, cluster in clustered_notes.items():
        print("Cluster ", i + 1)
        print(cluster)
        print("")


# ...
# Fast clustering
def fast_clustering(similarities, note_titles, min_community_size, threshold):
    """
    min_community_size: Only consider clusters that have at least 25 elements.
    threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar.
    """
    clusters = util.community_detection(similarities,
                                        min_community_size=min_community_size,
                                        threshold=threshold)
    # print(clusters)

    # Print for all clusters the top 3 and bottom 3 elements
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i + 1}, #{len(cluster)} Elements ")
        for note_id in cluster[0:3]:
            print("\t", note_titles[note_id])
        if len(cluster) > 3:
            print(f"\t ... and {len(cluster) - 3} more")
