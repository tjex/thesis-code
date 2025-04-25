#!/usr/bin/env python

# https://www.sbert.net/docs/quickstart.html

# check if correct conda env is active first
from sentence_transformers import SentenceTransformer
import clustering as c
import similarity as s
import corpus as cor
import numpy as np
import sys
import torch

stdin = sys.argv[:]

if len(stdin) < 2:
    print("No command provided.")
    print("Accepted commands:\n\ttrain\n\tsimdiss <note title>")
    exit(1)

corpus = cor.Corpus
model = SentenceTransformer("all-mpnet-base-v2")

similarities = torch.empty(0)
embeddings = np.array([])

# Set variables within class
corpus.init()

if stdin[1] == "train":

    print("Preparing corpus...")
    corpus.clean_notes()

    print("Generating embeddings...")
    corpus.generate_embeddings(model)
    embeddings = corpus.embeddings()

    print("Saving embeddings...")
    np.save("data/embeddings", embeddings)

    print("Calculating similarity scores (cosine)...")
    similarities = s.cos_sim_elementwise(embeddings)
    torch.save(similarities, "data/similarities.pt")

elif stdin[1] == "simdiss":
    title_input = stdin[2]
    # For title and note index lookup against user input.
    corpus.build_reference_data()

    print("Loading embeddings...")
    similarities = torch.load("data/similarities.pt")

    print(f"Calculating similarities against: {title_input}")
    c.note_simdiss(similarities, title_input)
