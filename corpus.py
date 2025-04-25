import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
import json, os


# Corpus works with the corpus of note data provided by zk.
class Corpus:

    @classmethod
    def init(cls):
        cls.note_data_dir = "data"
        cls.note_data = cls.note_data_dir + "/zk.json"
        cls.simdiss_results = os.path.join(cls.note_data_dir, "simdiss.json")

    @classmethod
    def clean_notes(cls):
        nltk.download("punkt")
        nltk.download("punkt_tab")
        nltk.download("stopwords")

        try:
            with open(cls.note_data) as f:
                notes = json.load(f)
        except:
            print(f"Could not open note data at: {cls.note_data}")
            exit(1)

        # taking the note body, and now raw, means frontmatter is not included.
        dirty_notes = [note["body"] for note in notes]
        stop_words = set(stopwords.words("english"))

        cleaned_notes = []

        for note in dirty_notes:
            tokens = sent_tokenize(note.lower())
            tokens = [
                t for t in tokens
                if t not in string.punctuation and t not in stop_words
            ]

            cleaned_notes.append(" ".join(tokens))

        cls.cleaned_notes = cleaned_notes

    @classmethod
    def build_reference_data(cls):
        with open(cls.note_data) as f:
            notes = json.load(f)

        # Extract desired fields from json
        titles = [title["title"] for title in notes]
        paths = [path["path"] for path in notes]

        paths_titles = []
        for i in range(0, len(titles)):
            title = titles[i]
            path = paths[i]
            combo = [title, path]
            paths_titles.append(combo)

        titles_dict = {}
        # create a dictionary for fast search
        for i, title in enumerate(titles):
            titles_dict[title] = i

        cls.paths_titles = paths_titles
        cls.titles_dict = titles_dict

    @classmethod
    def generate_embeddings(cls, model):
        cls.corpus_embeddings = model.encode(cls.cleaned_notes)

    # Getters

    # Return embeddings of entire corpus
    @classmethod
    def embeddings(cls):
        return cls.corpus_embeddings

    # Returns an array with all note paths and titles
    @classmethod
    def note_paths_titles(cls):
        return cls.paths_titles

    # Lookup the index of a note by title
    @classmethod
    def index_from_title(cls, title):
        return cls.titles_dict[title]
