import json
import regex as re


md_symbols_patt = r"(#)|(-)|(>)|(\*)|(\")"
md_link_patt = r"\[(.*?)\]\(.*?\)"


# Corpus works with the corpus of note data provided by zk.
class Corpus:
    @classmethod
    def init(cls):
        cls.note_data_dir = "data"
        cls.note_data = cls.note_data_dir + "/ps.json"

    @classmethod
    def prepare_corpus(cls):
        with open(cls.note_data) as f:
            notes = json.load(f)

        # Extract desired fields from json
        dirty_notes = [note["body"] for note in notes]
        titles = [title["title"] for title in notes]
        paths = [path["path"] for path in notes]

        paths_titles = []
        for i in range(0, len(titles)):
            title = titles[i]
            path = paths[i]
            combo = [title, path]
            paths_titles.append(combo)

        cleaned_notes = [""] * len(dirty_notes)

        for i, note in enumerate(dirty_notes):
            note = note.replace("\n", " ")  # flatten for regex ease of use
            # # NOTE: It may be better
            # to keep the links in the document, as links to the same document
            # should mean that relationships exist (as they've been defined by
            # the user themselves in the first place)
            # note = re.sub(md_link_patt, r"\1", note) 
            note = re.sub(md_symbols_patt, "", note)
            cleaned_notes[i] = note

        titles_dict = {}
        # create a dictionary for fast search
        for i, title in enumerate(titles):
            titles_dict[title] = i

        cls.cleaned_notes = cleaned_notes
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
