from bertopic import BERTopic
import pandas as pd
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
import os
from nltk.tokenize import sent_tokenize, word_tokenize

standard_stopwords = list(stopwords.words('english'))

model_dir = "data/bertopic"


class BTopic:

    @classmethod
    def init(cls, embedding_model, notes, titles, titles_dict):
        cls.embedding_model = embedding_model
        cls.notes = notes
        cls.titles = titles
        cls.titles_dict = titles_dict
        vectorizer = CountVectorizer(ngram_range=(2, 2),
                                     stop_words=standard_stopwords)

        cls.topic_model = BERTopic(top_n_words=10,
                                   n_gram_range=(2, 2),
                                   nr_topics=15,
                                   embedding_model=embedding_model,
                                   vectorizer_model=vectorizer,
                                   umap_model=UMAP(n_neighbors=15,
                                                   n_components=5,
                                                   min_dist=0.0,
                                                   metric='cosine'))

    @classmethod
    def _load_model(cls):
        cls.topic_model = cls.topic_model.load(model_dir, cls.embedding_model)
        return cls.topic_model

    @classmethod
    def _get_topic_data(cls):
        df = pd.read_pickle(os.path.join(model_dir, "topic_data.pkl"))
        return df

    # for testing implementations before setting an actual command
    @classmethod
    def misc(cls):
        print("misc")

    @classmethod
    def topic_search(cls, search_term):
        cls.topic_model = cls._load_model()
        topics, similarity = cls.topic_model.find_topics(search_term, top_n=5)
        print(topics, similarity)
        for t in topics:
            cls.topic_model.get_topic(t)

    # Code from:
    # https://towardsdatascience.com/topic-modelling-with-berttopic-in-python-8a80d529de34/
    @classmethod
    def derive_topics(cls):
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
            "prob": probabilities,
        })
        df.to_pickle(os.path.join(model_dir, "topic_data.pkl"))

    @classmethod
    def list_topics(cls):
        cls.topic_model = cls._load_model()
        topic_labels = cls.topic_model.generate_topic_labels(
            nr_words=3, topic_prefix=False, separator=" | ")
        topic_id = ""
        for i, t in enumerate(topic_labels):
            if i == 0:
                topic_id = "Outliers"
            else:
                topic_id = str(i - 1)

            print(f"{topic_id}: {t}")

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
        print(f"Documents topically related to: {title}\n")
        cls.list_docs_for_topic(note_topic)

    @classmethod
    def topic_vis(cls):
        cls.topic_model = cls._load_model()
        topic_1 = pd.DataFrame(data=cls.topic_model.get_topic(0),
                               columns=["Topic_1_word", "Topic_1_prob"])
        topic_2 = pd.DataFrame(data=cls.topic_model.get_topic(1),
                               columns=["Topic_2_word", "Topic_2_prob"])
        topic_3 = pd.DataFrame(data=cls.topic_model.get_topic(2),
                               columns=["Topic_3_word", "Topic_3_prob"])
        topic_4 = pd.DataFrame(data=cls.topic_model.get_topic(3),
                               columns=["Topic_4_word", "Topic_4_prob"])
        topic_5 = pd.DataFrame(data=cls.topic_model.get_topic(4),
                               columns=["Topic_5_word", "Topic_5_prob"])
        topic_6 = pd.DataFrame(data=cls.topic_model.get_topic(5),
                               columns=["Topic_6_word", "Topic_6_prob"])
        topics_df = pd.concat(
            [topic_1, topic_2, topic_3, topic_4, topic_5, topic_6], axis=1)

        # Initialize DataFrame to store reshaped data
        reshaped_data = pd.DataFrame(columns=['Bigram', 'Topic', 'Prob'])

        # Reshape data from original DataFrame
        for i in range(1, 7):
            topic_word_col = f'Topic_{i}_word'
            topic_prob_col = f'Topic_{i}_prob'
            temp_df = pd.DataFrame({
                'Bigram': topics_df[topic_word_col],
                'Topic': f'Topic {i}',
                'Prob': topics_df[topic_prob_col]
            })
            reshaped_data = pd.concat([reshaped_data, temp_df])

        # Set 'Bigram' as index
        reshaped_data.set_index('Bigram', inplace=True)

        # Create pivot table for heatmap
        pivot_table = reshaped_data.pivot(columns='Topic',
                                          values='Prob').fillna(0)

        # Plot heatmap
        plt.figure(figsize=(14, 14))
        cmap = sns.color_palette("crest", as_cmap=True)
        cmap.set_under(color='white')

        # Generate heatmap
        sns.heatmap(pivot_table,
                    cmap=cmap,
                    linewidths=0.5,
                    annot=False,
                    cbar=True,
                    mask=(pivot_table == 0),
                    vmin=0.0001)

        # Add bigrams with non-zero probability on top of the heatmap
        for i, bigram in enumerate(pivot_table.index):
            for j, topic in enumerate(pivot_table.columns):
                if pivot_table.loc[bigram, topic] > 0:
                    plt.text(j + 0.5,
                             i + 0.5,
                             bigram,
                             ha='center',
                             va='center',
                             color='black' if pivot_table.loc[bigram, topic]
                             > 0 else 'white',
                             fontsize=9)

        # Customize plot
        plt.title('Topic Modeling of Zettelkasten notes')
        plt.xlabel('Topic')
        plt.ylabel('Bigram')
        plt.tight_layout()
        plt.show()
