from sklearn.cluster import AgglomerativeClustering

# Agglomerative clustering. Returns clusters.
def agglo_clustering(embeddings, note_titles):
    clustering_model = AgglomerativeClustering(n_clusters=3, distance_threshold=None)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_notes = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_notes:
            clustered_notes[cluster_id] = []

        clustered_notes[cluster_id].append(note_titles[sentence_id])

    return clustered_notes
