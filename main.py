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

stdin = sys.argv[:]

if len(stdin) < 2:
    print("Accepted commands:\n\ttrain\n\tsimdiss <note title>")
    exit(1)


# nltk.download("punkt")

corpus = cor.Corpus
model = SentenceTransformer("all-mpnet-base-v2")

# Source, process and generate necessary data.
corpus.init()


if stdin[1] == "train":
    print("Preparing corpus...")
    corpus.clean_notes()

    print("Generating embeddings...")
    corpus.generate_embeddings(model)
    embeddings = corpus.embeddings()

    print("Saving embeddings...")
    np.save("data/embeddings", embeddings)

    # TODO: should calculate and save similarities here

elif stdin[1] == "simdiss":
    title_input = stdin[2]
    corpus.build_reference_data()

    print("Loading embeddings...")
    embeddings = np.load("data/embeddings.npy")

    print(f"Calculating similarities against: {title_input}")
    similarities = s.cos_sim_elementwise(embeddings)
    c.note_simdiss(similarities, title_input)

