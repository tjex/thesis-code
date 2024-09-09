from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import util
import corpus as cor

# https://www.sbert.net/examples/applications/clustering/README.html

corpus = cor.Corpus


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
# TODO: What should this function return for best usage with zk?
def note_simdiss(similarities, title):
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    s5 = []

    note_index = corpus.index_from_title(title)
    titles_arr = corpus.note_titles_array()

    # TODO: This should be its own function, where the user can define how many
    # segments the results should be split between.

    # cosine similarity returns a range of -1 to -1.
    mid = 0
    seg_length = 0.2
    seg1 = mid - (seg_length * 2)
    seg2 = mid - (seg_length)
    seg3 = mid + seg_length
    seg4 = mid + (seg_length * 2)

    for i, score in enumerate(similarities[note_index]):
        if i != note_index:
            t = titles_arr[i]
            s = similarities[note_index][i].item()
            s = round(s, 2)
            report = f"{t} ({str(s)})"

            if score <= seg1:
                s1.append(report)
            elif seg1 < score <= seg2:
                s2.append(report)
            elif seg2 < score <= seg3:
                s3.append(report)
            elif seg3 < score <= seg4:
                s4.append(report)
            elif seg4 < score:
                s5.append(report)

    # make sure all notes are included in the results (except the query note
    # iteslf, hence -1)
    print(
        f"DEBUG: {len(similarities[note_index]) - 1}, should equal {len(s1) + len(s2) + len(s3) + len(s4) + len(s5)}"
    )
    print(f"mean: {mid}\nseg1: {seg1}\nseg2 {seg2}\nseg3 {seg3}\nseg4 {seg4}\n")

    full_output = True
    if full_output:
        print(f"Similarity relationships for, {titles_arr[note_index]}")
        print(f"Least similar, {len(s1)} notes:")
        print("\t", "\n\t".join([note for note in s1]))
        print()
        print(f"Slightly similar, {len(s2)} notes:")
        print("\t", "\n\t".join([note for note in s2]))
        print()
        print(f"Somewhat similar, {len(s3)} notes:")
        print("\t", "\n\t".join([note for note in s3]))
        print()
        print(f"Very similar, {len(s4)} notes:")
        print("\t", "\n\t".join([note for note in s4]))
        print()
        print(f"Most similar, {len(s5)} notes:")
        print("\t", "\n\t".join([note for note in s5]))
