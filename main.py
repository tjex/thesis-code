#!/usr/bin/env python

# https://www.sbert.net/docs/quickstart.html

# check if correct conda env is active first
from sentence_transformers import SentenceTransformer
import nltk
import clustering as c
import similarity as s
import corpus as cor
import numpy as np
import sys

title_input = sys.argv[1]

nltk.download("punkt")
corpus = cor.Corpus
model = SentenceTransformer("all-mpnet-base-v2")

# Source, process and generate necessary data.
corpus.init()
corpus.prepare_corpus()
corpus.generate_embeddings(model)
embeddings = corpus.embeddings()
np.save("data/embeddings", embeddings)
embeddings = np.load("data/embeddings.npy")
similarities = s.cos_sim_elementwise(embeddings)

# Similarity
c.note_simdiss(similarities, title_input)
