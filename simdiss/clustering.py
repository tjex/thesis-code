from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import util
import torch

# https://www.sbert.net/examples/applications/clustering/README.html


# Agglomerative clustering. Returns clusters.
def agglo_clustering(similarities, note_titles, nc):
    clustering_model = AgglomerativeClustering(n_clusters=nc, distance_threshold=None)
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


# Fast clustering
def fast_clustering(similarities, note_titles):
    # Two parameters to tune:
    # min_cluster_size: Only consider cluster that have at least 25 elements
    # threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
    clusters = util.community_detection(
        similarities, min_community_size=3, threshold=0.3
    )
    print(clusters)

    # Print for all clusters the top 3 and bottom 3 elements
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i + 1}, #{len(cluster)} Elements ")
        for note_id in cluster[0:3]:
            print("\t", note_titles[note_id])
        if len(cluster) > 3:
            print(f"\t ... and {len(cluster) - 3} more")


# for each note, find the most and least similar notes
def manual(similarities, note_titles, note_index):
    s1 = []
    s2 = []
    s3 = []
    s4 = []

    sum = 0
    mean = 0
    for val in similarities[note_index]:
        if val < 1:
            sum += val
        mean = sum / len(similarities[note_index])

    # throw error here
    if mean == 0:
        return

    seg_length = mean / 20
    seg1 = seg_length
    seg2 = seg_length * 2
    seg3 = seg_length * 3
    seg4 = seg_length * 4

    print(seg1, seg2, seg3, seg4)

    for i, score in enumerate(similarities[note_index]):
        if score <= seg1:
            s1.append(note_titles[i])
        elif score > seg1 <= seg2:
            s2.append(note_titles[i])
        elif score > seg3 <= seg4:
            s3.append(note_titles[i])
        elif score > seg4 < 1:
            s4.append(note_titles[i])

    print(f"DEBUG: {len(similarities[note_index])}, should equal {len(s1) + len(s2) + len(s3) + len(s4)}")

    print(f"Similarity relationships for, {note_titles[note_index]}")
    print(f"Least similar, {len(s1)} notes:")
    # print("\t", "\n\t".join([note for note in s1]))
    print()
    print(f"Somewhat similar, {len(s2)} notes:")
    # print("\t", "\n\t".join([note for note in s2]))
    print()
    print(f"Very similar, {len(s3)} notes:")
    # print("\t", "\n\t".join([note for note in s3]))
    print()
    print(f"Most similar, {len(s4)} notes:")
    # print("\t", "\n\t".join([note for note in s4]))
