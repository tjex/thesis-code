#!/usr/bin/env python

# https://www.sbert.net/docs/quickstart.html

from sentence_transformers import SentenceTransformer
import topic_modelling
import similarity
import corpus as cor
import argparse
import util


def cli_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # simdiss args
    simdiss_parser = subparsers.add_parser(
        "sl", help="Compare note similarity against corpus")
    simdiss_subparsers = simdiss_parser.add_subparsers(dest="simdiss_command",
                                                       required=True)
    simdiss_subparsers.add_parser("train",
                                  help="Train similarity learning model.")

    cluster_parser = simdiss_subparsers.add_parser(
        "cluster", help="Cluster notes by similarity.")
    cluster_parser.add_argument("--clusters",
                                help="Number of clusters.",
                                type=int,
                                default=10)

    compare_parser = simdiss_subparsers.add_parser(
        "compare",
        help="Compare similarities of all notes against given note.")
    compare_parser.add_argument("--title", help="Title of note to compare.")
    compare_parser.add_argument(
        "--strategy",
        help="Strategy to group notes: std (i.e, standard deviation) or even.",
        type=str,
        default="std")

    # topic modelling args
    topic_parser = subparsers.add_parser("tm",
                                         help="Work with topic modelling.")
    topic_subparsers = topic_parser.add_subparsers(dest="topic_command")

    search_parser = topic_subparsers.add_parser(
        "search", help="Find topics most similar to search term.")
    search_parser.add_argument("term", type=str, help="Topic term to search.")

    topic_subparsers.add_parser("train", help="Train topic model.")

    list_parser = topic_subparsers.add_parser("list",
                                              help="List various results.")
    list_parser.add_argument("--topics",
                             action="store_true",
                             help="List topics")
    list_parser.add_argument("--docs-for-topic",
                             type=int,
                             help="List documents belonging to a given topic.")
    list_parser.add_argument(
        "--related",
        type=str,
        help="List other notes that share the same topic.")

    args = parser.parse_args()
    return args


def main():
    args = cli_args()

    model = SentenceTransformer("all-mpnet-base-v2")
    model.max_seq_length = 500

    # Prepare data for downstream tasks
    corpus = cor.Corpus
    corpus.init()
    corpus.clean_notes()
    corpus.build_reference_data()

    # Init SBERT and BERTopic classes
    sbert = similarity.SBERT
    bertopic = topic_modelling.BTopic
    bertopic.init(model, corpus.cleaned_notes, corpus.titles,
                  corpus.titles_dict)

    if args.command == "sl":
        match args.simdiss_command:
            case "train":
                embeddings = sbert.generate_embeddings(model,
                                                       corpus.cleaned_notes)
                similarities = similarity.cos_sim_elementwise(embeddings)
                util.save_similarities(similarities)
                similarity.least_similar_note(similarities)

            case "compare":
                title_input = args.title
                similarities = util.load_similarities()
                similarity.note_simdiss(similarities,
                                        title_input,
                                        strategy=args.strategy)

            case "cluster":
                sbert.agglo_clustering(args.clusters)

    if args.command == "tm":
        match args.topic_command:
            case "search":
                bertopic.topic_search(args.term)

            case "train":
                bertopic.derive_topics()

            case "list":
                if args.topics:
                    bertopic.list_topics()

                if args.docs_for_topic is not None:
                    bertopic.list_docs_for_topic(args.docs_for_topic)

                if args.related is not None:
                    bertopic.list_topically_related_notes(args.related)


if __name__ == "__main__":
    main()
