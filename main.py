#!/usr/bin/env python

# https://www.sbert.net/docs/quickstart.html

import bertopic
from sentence_transformers import SentenceTransformer
import topic_modeling
import clustering
import similarity
import corpus as cor
import numpy as np
import torch
import argparse


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Generate embeddings and similarity matrix")
    train_parser.add_argument("--sl",
                              action="store_true",
                              help="Train similarity learning model.")
    train_parser.add_argument("--tm",
                              action="store_true",
                              help="Train topic model.")

    simdiss_parser = subparsers.add_parser(
        "simdiss", help="Compare note similarity against corpus")
    simdiss_parser.add_argument("title", help="Title of note to compare")
    simdiss_parser.add_argument(
        "--strategy",
        help="Strategy to group notes: std (i.e, standard deviation) or even",
        default="std")

    subparsers.add_parser("cluster", help="Cluster notes by similarity.")
    subparsers.add_parser("topic-vis",
                          help="Perform topic modelling on notes.")

    args = parser.parse_args()

    model = SentenceTransformer("all-mpnet-base-v2")
    model.max_seq_length = 500

    similarities = torch.empty(0)
    embeddings = np.array([])

    # Init corpus / data sources
    corpus = cor.Corpus()
    corpus.init()
    corpus.clean_notes()
    corpus.build_reference_data()

    # Init BERTopic
    bertopic = topic_modeling.BTopic()
    notes = corpus.cleaned_notes
    bertopic.init(model, notes)

    if args.command == "train":
        if args.sl:
            print("Generating embeddings...")
            corpus.generate_embeddings(model)
            embeddings = corpus.embeddings()

            print("Saving embeddings...")
            np.save("data/embeddings", embeddings)

            print("Calculating similarity scores (cosine)...")
            similarities = similarity.cos_sim_elementwise(embeddings)
            torch.save(similarities, "data/similarities.pt")

        if args.tm:
            print("Generating topic model...")
            bertopic.derive_topics()

    elif args.command == "simdiss":
        title_input = args.title

        print("Loading embeddings...")
        similarities = torch.load("data/similarities.pt")

        print(f"Calculating similarities against: {title_input}")
        similarity.note_simdiss(similarities,
                                title_input,
                                strategy=args.strategy)

    elif args.command == "cluster":
        print("Loading embeddings...")
        similarities = torch.load("data/similarities.pt")
        clustering.agglo_clustering(similarities, corpus.titles, 5)
        clustering.fast_clustering(similarities, corpus.titles, 15, 0.95)

    elif args.command == "topic-vis":
        print("Running topic modelling with BERTopic...")
        bertopic.topic_vis()


if __name__ == "__main__":
    main()
