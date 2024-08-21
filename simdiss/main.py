# https://www.sbert.net/docs/quickstart.html

# check if correct conda env is active first
import json
import re
import torch
import clustering as c

from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from sentence_transformers import SentenceTransformer


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

# EMBEDDING

embeddings = model.encode(cleaned_notes)

# SIMILARITY

similarities = model.similarity(embeddings, embeddings)
print("similarities")
print(similarities)

dissimilarities = similarities
for i, row in enumerate(dissimilarities.numpy()):
    for j, val in enumerate(row):
        if i != j:
            row[j] = 1 - val

print("dissimilarities")
print(torch.round(dissimilarities, decimals=4))

# # example similarity output
# note1 = 10
# note2 = 20
# print()
# print(
#     "Similarity score between:\n",
#     "-",
#     note_titles[note1],
#     "\n",
#     "-",
#     note_titles[note2],
#     "\n\n",
#     similarities[note1][note2],
# )

# CLUSTERING
# c.agglo_clustering(similarities, note_titles, 10)
# print()
# c.fast_clustering(embeddings, note_titles)
