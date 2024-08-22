# https://www.sbert.net/docs/quickstart.html

# check if correct conda env is active first
import clustering as c
import similarity as s
import parse as p

from sentence_transformers import SentenceTransformer

# nltk.download("punkt")

# model = SentenceTransformer("sentence-t5-base")
model = SentenceTransformer("all-MiniLM-L6-v2")

cleaned_notes, note_titles = p.corpus("data/ps.json")
embeddings = model.encode(cleaned_notes)

# SIMILARITY

similarities = s.cos_sim(embeddings)

c.note_simdiss(similarities, note_titles, 51)


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
# c.agglo_clustering(similarities, note_titles, 6)
# print("---------------------------------")
# print("---------------------------------")
# c.agglo_clustering(dissimilarities, note_titles, 6)
# print()
# c.fast_clustering(embeddings, note_titles)
