# https://www.sbert.net/docs/quickstart.html

# check if correct conda env is active first
import clustering as c
import similarity as s
import corpus as cor

# nltk.download("punkt")
corpus = cor.Corpus

# Source, process and generate necessary data.
corpus.init("data/100_clones.json")
corpus.prepare_corpus()
corpus.generate_embeddings()
embeddings = corpus.embeddings()
similarities = s.cos_sim(embeddings)
print(similarities)

# Similarity
title = "foo-5"
c.note_simdiss(similarities, title)

# Clustering
# c.agglo_clustering(similarities, note_titles, 6)
# print("---------------------------------")
# print("---------------------------------")
# c.agglo_clustering(dissimilarities, note_titles, 6)
# print()
# c.fast_clustering(embeddings, note_titles)
