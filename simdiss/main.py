# https://www.sbert.net/docs/quickstart.html

# check if correct conda env is active first
import clustering as c
import similarity as s
import note_data

from sentence_transformers import SentenceTransformer

# nltk.download("punkt")

# model = SentenceTransformer("sentence-t5-base")
model = SentenceTransformer("all-MiniLM-L6-v2")

nd = note_data.NoteData
nd.corpus("data/ps.json")
cleaned_notes = nd.note_bodies()

embeddings = model.encode(cleaned_notes)

# SIMILARITY

similarities = s.cos_sim(embeddings)

title = "YUA - How will AI affect the 2024 elections?"
c.note_simdiss(similarities, title)


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
