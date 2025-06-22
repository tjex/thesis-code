from sentence_transformers import util as sbert_utils
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import torch, json
from corpus import Corpus as corpus
import util

class SBERT:
    @classmethod
    def generate_embeddings(cls, model, notes):
        print("Generating embeddings...")
        embeddings = model.encode(notes)
        util.save_embeddings(embeddings)
        return embeddings

    # Cluster similar notes together
    # Agglomerative clustering.
    @classmethod
    def agglo_clustering(cls, nc):
        """
        nc: number of clusters to divide into.
        """
        similarities = util.load_similarities()
        note_titles = corpus.titles
        clustering_model = AgglomerativeClustering(n_clusters=nc,
                                                   distance_threshold=None)
        clustering_model.fit(similarities)
        cluster_assignment = clustering_model.labels_

        clustered_notes = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            if cluster_id not in clustered_notes:
                clustered_notes[cluster_id] = []

            clustered_notes[cluster_id].append(note_titles[sentence_id])

        for i, cluster in sorted(clustered_notes.items()):
            print(f"Cluster {i}:")
            for j, title in enumerate(cluster):
                print(f"{j}. {title}")
            print("")

# The max value of a similarity tensor is always 1.0
# this will skew subsequent calculations such as averages and other ranges.
# Removing the incoming notes index itself removes this skew.
def unbiased_min_max(tensor, note_index) -> tuple[float, float]:
    t = torch.cat((tensor[:note_index], tensor[note_index + 1:]))

    min = torch.min(t).item()
    max = torch.max(t).item()

    return min, max

# Get the least similar note from the corpus
# Useful for testing ability of similarity learning and also 
# for the sake of interest!
def least_similar_note(similarities):

    # Remove self-similarities (diagonal entries)
    t_no_diag = similarities.clone()
    t_no_diag.fill_diagonal_(0)

    total_similarity = t_no_diag.sum(dim=1)

    index = torch.argmin(total_similarity)
    title = corpus.titles[index]
    print(title)

# Modified code of SentenceTransformers.util.cos_sim().
# This proves a [note x note] similarity matrix, which can be
# later queried against.
# The default implementation compairs similarity scores of two
# inputs: https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
def cos_sim_elementwise(embeddings):
    print("Calculating similarity scores (cosine)...")
    a = sbert_utils._convert_to_batch_tensor(embeddings)
    a_norm = sbert_utils.normalize_embeddings(a)
    return torch.mm(a_norm, a_norm.transpose(0, 1))


# creates four values that are used as cross-over / division points
# for clustering similarity results into "least similar" to "most similar"
# segments
def even_divisions(min, max):
    mid = (min + max) / 2
    sim_range = max - min
    seg_length = sim_range / 6
    div1 = round(mid - (seg_length * 2), 4)
    div2 = round(mid - (seg_length), 4)
    div3 = round(mid + seg_length, 4)
    div4 = round(mid + (seg_length * 2), 4)

    return div1, div2, div3, div4


def std_dev_divisions(similarities):
    similarities = np.array(similarities)
    mean = np.mean(similarities)
    std = np.std(similarities)

    div1 = round(mean - 2 * std, 2)
    div2 = round(mean - 1 * std, 2)
    div3 = round(mean + 1 * std, 2)
    div4 = round(mean + 2 * std, 2)

    return div1, div2, div3, div4


# Process a simdiss for a singular note against all other notes
# in the corpus. There are 5 segments from "least similar" to "most similar".
def note_simdiss(similarities, title, strategy):
    print(f"Calculating similarities against: {title}")
    # s for segments. For "least similar" to "most similar" notes.
    s1, s2, s3, s4, s5 = [], [], [], [], []

    try:
        note_index = corpus.get_index_from_title(title)
    except:
        print(f"Error retrieving index by title: {title}")
        print(
            f"The note either does not exist, or the note data does not contain this note."
        )
        exit(1)

    titles_paths = list(zip(corpus.titles, corpus.paths))

    # Retrieve similarity scores for the incoming note only
    similarities = similarities[note_index]

    if strategy == "even":
        min, max = unbiased_min_max(similarities, note_index)
        div1, div2, div3, div4 = even_divisions(min, max)
    else:
        div1, div2, div3, div4 = std_dev_divisions(similarities)

    for i, score in enumerate(similarities):
        if i != note_index:
            sim = round(score.item(), 2)
            element = [titles_paths[i][0], titles_paths[i][1], sim]

            if score <= div1:
                # order: least similar to most similar
                s1.append(element)
            elif div1 < score <= div2:
                s2.append(element)
            elif div2 < score <= div3:
                s3.append(element)
            elif div3 < score <= div4:
                s4.append(element)
            elif div4 < score:
                s5.append(element)

    path = corpus.paths[note_index]
    build_json_file(title, path, s1, s2, s3, s4, s5)


def build_json_file(note_title, note_path, s1, s2, s3, s4, s5):

    # Sort by similarity score
    for group in [s1, s2, s3, s4, s5]:
        group.sort(key=lambda x: x[2])

    json_data = {
        "title":
        note_title,
        "path":
        note_path,
        "least_similar": [{
            "title": title,
            "path": path,
            "similarity": sim
        } for title, path, sim in s1],
        "somewhat_similar": [{
            "title": title,
            "path": path,
            "similarity": sim
        } for title, path, sim in s2],
        "moderately_similar": [{
            "title": title,
            "path": path,
            "similarity": sim
        } for title, path, sim in s3],
        "very_similar": [{
            "title": title,
            "path": path,
            "similarity": sim
        } for title, path, sim in s4],
        "most_similar": [{
            "title": title,
            "path": path,
            "similarity": sim
        } for title, path, sim in s5],
    }

    try:
        with open(corpus.simdiss_results, "w") as file:
            json.dump(json_data, file)
    except:
        print(f"Could not write file {corpus.simdiss_results}")
    else:
        print(f"Results saved to {corpus.simdiss_results}")
