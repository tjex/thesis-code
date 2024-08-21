from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import util

# https://www.sbert.net/examples/applications/clustering/README.html


# Agglomerative clustering. Returns clusters.
def agglo_clustering(embeddings, note_titles, nc):
    clustering_model = AgglomerativeClustering(
        n_clusters=nc, distance_threshold=None
    )
    clustering_model.fit(embeddings)
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


# Fast clustering
def fast_clustering(embeddings, note_titles):
    # Two parameters to tune:
    # min_cluster_size: Only consider cluster that have at least 25 elements
    # threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
    clusters = util.community_detection(embeddings, min_community_size=3, threshold=0.3)
    print(clusters)

    # Print for all clusters the top 3 and bottom 3 elements
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i + 1}, #{len(cluster)} Elements ")
        for note_id in cluster[0:3]:
            print("\t", note_titles[note_id])
        if len(cluster) > 3:
            print(f"\t ... and {len(cluster) - 3} more")
