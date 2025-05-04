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

    # simdiss args
    simdiss_parser = subparsers.add_parser(
        "sl", help="Compare note similarity against corpus")
    simdiss_subparsers = simdiss_parser.add_subparsers(dest="simdiss_command",
                                                       required=True)
    simdiss_subparsers.add_parser(
        "train", help="Generate embeddings and similarity matrix.")

    simdiss_subparsers.add_parser("cluster",
                                  help="Cluster notes by similarity.")

    compare_parser = simdiss_subparsers.add_parser(
        "compare",
        help="Compare similarities of all notes against given note.")
    compare_parser.add_argument("--title", help="Title of note to compare.")
    compare_parser.add_argument(
        "--strategy",
        help="Strategy to group notes: std (i.e, standard deviation) or even.",
        default="std")

    # topic modeling args
    topic_parser = subparsers.add_parser("tm",
                                         help="Work with topic modeling.")
    topic_subparsers = topic_parser.add_subparsers(dest="topic_command",
                                                   required=True)

    topic_subparsers.add_parser("train", help="Train topic model.")
    topic_subparsers.add_parser("list", help="List topics.")
    topic_subparsers.add_parser("misc", help="For testing.")

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

    if args.command == "sl":
        match args.simdiss_command:
            case "train":
                print("Generating embeddings...")
                corpus.generate_embeddings(model)
                embeddings = corpus.embeddings()

                print("Saving embeddings...")
                np.save("data/embeddings", embeddings)

                print("Calculating similarity scores (cosine)...")
                similarities = similarity.cos_sim_elementwise(embeddings)
                torch.save(similarities, "data/similarities.pt")

            case "compare":
                title_input = args.title

                print("Loading embeddings...")
                similarities = torch.load("data/similarities.pt")

                print(f"Calculating similarities against: {title_input}")
                similarity.note_simdiss(similarities,
                                        title_input,
                                        strategy=args.strategy)

            case "cluster":
                print("Loading embeddings...")
                similarities = torch.load("data/similarities.pt")
                clustering.agglo_clustering(similarities, corpus.titles, 5)
                clustering.fast_clustering(similarities, corpus.titles, 15,
                                           0.95)

    if args.command == "tm":
        match args.topic_command:
            case "train":
                print("Generating topic model...")
                bertopic.derive_topics()

            case "topic-vis":
                bertopic.topic_vis()

            case "list":
                bertopic.document_topics()

            case "misc":
                docs = bertopic.docs_for_topic(1)
                print(docs)


if __name__ == "__main__":
    main()
