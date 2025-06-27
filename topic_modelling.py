from bertopic import BERTopic
import pandas as pd
from umap import UMAP
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import os
import corpus
import util

standard_stopwords = list(stopwords.words('english'))

model_dir = "data/bertopic"


class BTopic:

    @classmethod
    def init(cls, embedding_model, notes, titles, titles_dict):
        cls.embedding_model = embedding_model
        cls.notes = notes
        cls.titles = titles
        cls.titles_dict = titles_dict
        vectorizer_model = CountVectorizer(ngram_range=(1, 2),
                                           min_df=2,
                                           stop_words=standard_stopwords)

        cls.topic_model = BERTopic(top_n_words=10,
                                   n_gram_range=(1, 2),
                                   min_topic_size=8,
                                   nr_topics="auto",
                                   embedding_model=embedding_model,
                                   vectorizer_model=vectorizer_model,
                                   umap_model=UMAP(n_neighbors=10,
                                                   n_components=5,
                                                   min_dist=0.0,
                                                   metric='cosine', 
                                                   random_state=42))

    @classmethod
    def _load_model(cls) -> BERTopic:
        cls.topic_model = cls.topic_model.load(model_dir, cls.embedding_model)
        return cls.topic_model

    @classmethod
    def _get_topic_data(cls):
        df = pd.read_pickle(os.path.join(model_dir, "topic_data.pkl"))
        return df

    # Code from:
    # https://towardsdatascience.com/topic-modelling-with-berttopic-in-python-8a80d529de34/
    @classmethod
    def derive_topics(cls):
        print("Generating topic model...")
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        topics, probabilities = cls.topic_model.fit_transform(cls.notes)
        cls.topic_model.save(model_dir,
                             serialization="safetensors",
                             save_ctfidf=True,
                             save_embedding_model=cls.embedding_model)

        df = pd.DataFrame({
            "topic": topics,
            "title": cls.titles,
            "doc": cls.notes,
            "prob": probabilities
        })
        df.to_pickle(os.path.join(model_dir, "topic_data.pkl"))

    @classmethod
    def topic_search(cls, search_term):
        """
        Return the top three topics that are related to the search term.
        Prints the topics, topic labels and their descriptive score.
        """
        cls.topic_model = cls._load_model()
        topics, similarity = cls.topic_model.find_topics(search_term, top_n=3)

        for i, t in enumerate(topics):
            indi_topics = cls.topic_model.get_topic(t)
            sim = round(similarity[i], 2) * 100
            print("\n")
            print(f"Topic {t} ({sim}% similar to '{search_term}')")
            for j, it in enumerate(indi_topics):
                if j >= 4:
                    # Limit the amount of topic labels
                    continue
                print(f"{it[0]}")

    @classmethod
    def list_topics(cls):
        cls.topic_model = cls._load_model()

        topic_labels = cls.topic_model.generate_topic_labels(
            nr_words=3, topic_prefix=False, separator=", ")
        topic_id = ""

        print("Schema:")
        print(f"[Topic ID: (top three topic labels)]\n")

        for i, t in enumerate(topic_labels):
            if i == 0:
                topic_id = "Outliers"
            else:
                topic_id = str(i - 1)

            print(f"{topic_id}: ({t})")

    @classmethod
    def list_docs_for_topic(cls, topic_id):
        # get_representitive_documents() could be used here, but it requires the
        # model.fit_transform() and therefore for the fitted model to be loaded
        # in ram. Doing it manually here as is more efficient.
        df = cls._get_topic_data()

        topic_docs = df[df["topic"] == topic_id]

        if topic_docs.empty:
            print(f"No documents found for topic {topic_id}.")
        else:
            for i, doc in enumerate(topic_docs["title"], 1):
                print(f"{i}. {doc}")

    @classmethod
    def list_topically_related_notes(cls, title):
        note_index = cls.titles_dict[title]
        df = cls._get_topic_data()
        note_topic = df["topic"][note_index]
        print(f"Notes topically related to, {title}")
        cls.list_docs_for_topic(note_topic)
