# https://www.sbert.net/docs/quickstart.html

# check if correct conda env is active first
import json
import re
import torch

import nltk
from nltk import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering


# nltk.download("punkt")

# model = SentenceTransformer("sentence-t5-base")
model = SentenceTransformer("all-MiniLM-L6-v2")

wd = TreebankWordDetokenizer()
wt = TreebankWordTokenizer()

md_symbols_patt = r"(#)|(-)|(>)|(\*)|(\")"
md_link_patt = r"\[(.*?)\]\(.*?\)"

## get an clean data

with open("data/ps.json") as f:
    notes = json.load(f)


dirty_notes = [note["body"] for note in notes]
note_titles = [title["title"] for title in notes]

cleaned_notes = [""] * len(dirty_notes)

for i, note in enumerate(dirty_notes):
    note = note.replace("\n", " ")  # flatten for regex ease of use
    note = re.sub(md_link_patt, r"\1", note)
    note = re.sub(md_symbols_patt, "", note)
    cleaned_notes[i] = note
    # note_toks += sent_tokenize(note)
    # note_toks = dtkn.detokenize(note_toks)
    # note_body_toks += note_toks


# write to file
# with open("data/cleaned_notes.txt", "w") as cleaned_notes_file:
#     for n in cleaned_notes:
#         cleaned_notes_file.write(n)
#         cleaned_notes_file.write("\n\n")


# EMBEDDING

embeddings = model.encode(cleaned_notes)

# SIMILARITY

similarities = model.similarity(embeddings, embeddings)
# min, _ = similarities.min(dim=0, keepdim=False)
# minmax = similarities.aminmax(dim=0, keepdim=False)

# example similarity output
note1 = 10
note2 = 20
print()
print(
    "Similarity score between:\n",
    "-",
    note_titles[note1],
    "\n",
    "-",
    note_titles[note2],
    "\n\n",
    similarities[note1][note2],
)

# CLUSERING

# Perform agglomerative clustering
clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=2)
clustering_model.fit(embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []

    clustered_sentences[cluster_id].append(note_titles[sentence_id])

for i, cluster in clustered_sentences.items():
    print("Cluster ", i + 1)
    print(cluster)
    print("")
