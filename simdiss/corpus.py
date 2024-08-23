import json
import regex as re
from sentence_transformers import SentenceTransformer


md_symbols_patt = r"(#)|(-)|(>)|(\*)|(\")"
md_link_patt = r"\[(.*?)\]\(.*?\)"


# Corpus works with the corpus of note data provided by zk.
class Corpus:
    @classmethod
    def init(cls, path):
        cls.note_data = path
        cls.model = SentenceTransformer("all-MiniLM-L6-v2")

    @classmethod
    def prepare_corpus(cls):
        with open(cls.note_data) as f:
            notes = json.load(f)

        # Extract desired fields from json
        dirty_notes = [note["body"] for note in notes]
        titles_array = [title["title"] for title in notes]

        titles_dict = {}
        # create a dictionary for fast search
        for i, title in enumerate(titles_array):
            titles_dict[title] = i

        cleaned_notes = [""] * len(dirty_notes)

        for i, note in enumerate(dirty_notes):
            note = note.replace("\n", " ")  # flatten for regex ease of use
            note = re.sub(md_link_patt, r"\1", note)
            note = re.sub(md_symbols_patt, "", note)
            cleaned_notes[i] = note

        cls.cleaned_notes = cleaned_notes
        cls.titles_array = titles_array
        cls.titles_dict = titles_dict

    @classmethod
    def generate_embeddings(cls):
        cls.corpus_embeddings = cls.model.encode(cls.cleaned_notes)

    # Getters

    # Return embeddings of entire corpus
    @classmethod
    def embeddings(cls):
        return cls.corpus_embeddings

    # Returns an array with all note titles
    @classmethod
    def note_titles_array(cls):
        return cls.titles_array

    # Returns a dictionary with form [title]: i
    @classmethod
    def note_titles_dict(cls):
        return cls.titles_dict

    # Lookup the index of a note by title
    @classmethod
    def index_from_title(cls, title):
        return cls.titles_dict[title]
