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

    topic_subparsers.add_parser("misc", help="For testing.")
    topic_subparsers.add_parser("train", help="Train topic model.")
    list_parser = topic_subparsers.add_parser("list",
                                              help="List various results.")
    list_parser.add_argument("--topics",
                             action="store_true",
                             help="List topics")
    list_parser.add_argument("--topic-search",
                             type=str,
                             help="Find topics most similar to search term.")
    list_parser.add_argument(
        "--topically-related",
        type=str,
        help="List other notes that share the same topic.")
    list_parser.add_argument("--docs-for-topic", type=int, help="List topics.")

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
    bertopic.init(model, corpus.cleaned_notes, corpus.titles,
                  corpus.titles_dict)

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
                if args.topics:
                    bertopic.list_topics()
                if args.topic_search is not None:
                    bertopic.topic_search(args.topic_search)
                if args.docs_for_topic is not None:
                    print(f"Documents for topic {args.docs_for_topic}:\n")
                    bertopic.list_docs_for_topic(args.docs_for_topic)
                if args.topically_related is not None:
                    bertopic.list_topically_related_notes(
                        args.topically_related)

            case "misc":
                bertopic.misc()


if __name__ == "__main__":
    main()
