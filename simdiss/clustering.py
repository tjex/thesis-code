from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import util
import util as u
import corpus as c

# https://www.sbert.net/examples/applications/clustering/README.html

corpus = c.Corpus


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


# creates four values that are used as cross-over / division points
# for clustering similarity results into "least similar" to "most similar"
# segments
def calculate_divisions(min, max):
    mid = (min + max) / 2
    sim_range = max - min
    seg_length = sim_range / 6
    div1 = round(mid - (seg_length * 2), 4)
    div2 = round(mid - (seg_length), 4)
    div3 = round(mid + seg_length, 4)
    div4 = round(mid + (seg_length * 2), 4)

    return div1, div2, div3, div4


# Process a simdiss for a singular note against all other notes
# in the corpus. There are 5 segments from "least similar" to "most similar".
# TODO: What should this function return for best usage with zk?
def note_simdiss(similarities, title):
    # s for segments. For "least similar" to "most similar" notes.
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    s5 = []

    note_index = corpus.index_from_title(title)
    titles_arr = corpus.note_titles_array()
    # reduce to a 1d array for simplified handling
    similarities = similarities[note_index]

    min, max = u.unbiased_min_max(similarities, note_index)
    div1, div2, div3, div4 = calculate_divisions(min, max)

    for i, score in enumerate(similarities):
        if i != note_index:
            t = titles_arr[i]
            s = round(similarities[i].item(), 2)
            report = f"{t} ({str(s)})"

            if score <= div1:
                s1.append(report)
            elif div1 < score <= div2:
                s2.append(report)
            elif div2 < score <= div3:
                s3.append(report)
            elif div3 < score <= div4:
                s4.append(report)
            elif div4 < score:
                s5.append(report)

    print(
        f"""
        DEBUG:\n
        min: {min}
        max: {max}
        div1: {div1}
        div2: {div2}
        div3: {div3}
        div4: {div4}
        """
    )

    # TODO: this should just be a test case.
    if len(similarities) - 1 != len(s1) + len(s2) + len(s3) + len(s4) + len(s5):
        print(
            "Bug alert. Length of similarities array does not equal sum of notes in all segments."
        )
        exit()

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
