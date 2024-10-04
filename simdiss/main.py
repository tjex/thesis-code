#!/usr/bin/env python

# https://www.sbert.net/docs/quickstart.html

# check if correct conda env is active first
from sentence_transformers import SentenceTransformer
import clustering as c
import similarity as s
import corpus as cor
import sys

# nltk.download("punkt")
corpus = cor.Corpus
model = SentenceTransformer("all-mpnet-base-v2")


# Source, process and generate necessary data.
# NOTE: As these scripts are executed by main.go, paths need to be relative from
# main.go, and not main.py!
corpus.init("./simdiss/data/ps.json")
corpus.prepare_corpus()
corpus.generate_embeddings(model)
embeddings = corpus.embeddings()
similarities = s.cos_sim_elementwise(embeddings)

# Similarity
# title = "The Academic Paper - Abstract"
title = "what defines the Zettelkasten system?"
c.note_simdiss(similarities, title)

# Clustering
# c.agglo_clustering(similarities, note_titles, 6)
# print("---------------------------------")
# print("---------------------------------")
# c.agglo_clustering(dissimilarities, note_titles, 6)
# print()
# c.fast_clustering(embeddings, note_titles)
