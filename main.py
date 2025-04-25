#!/usr/bin/env python

# https://www.sbert.net/docs/quickstart.html

from sentence_transformers import SentenceTransformer
import clustering as c
import similarity as s
import corpus as cor
import numpy as np
import torch
import argparse


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train",
                          help="Generate embeddings and similarity matrix")
    simdiss_parser = subparsers.add_parser(
        "simdiss", help="Compare note similarity against corpus")
    simdiss_parser.add_argument("title", help="Title of note to compare")

    args = parser.parse_args()

    corpus = cor.Corpus
    model = SentenceTransformer("all-mpnet-base-v2")

    similarities = torch.empty(0)
    embeddings = np.array([])

    # Set variables within class
    corpus.init()

    if args.command == "train":

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

    elif args.command == "simdiss":
        title_input = args.title
        # For title and note index lookup against user input.
        corpus.build_reference_data()

        print("Loading embeddings...")
        similarities = torch.load("data/similarities.pt")

        print(f"Calculating similarities against: {title_input}")
        c.note_simdiss(similarities, title_input)


if __name__ == "__main__":
    main()
