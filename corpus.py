import json, os
import re

md_symbols_patt = r"(#)|(>)|(\*)|(_)"
md_link_patt = r"\[(.*?)\]\(.*?\)"


# Corpus works with the corpus of note data provided by zk.
class Corpus:

    @classmethod
    def init(cls):
        cls.note_data_dir = "data"
        cls.note_data = cls.note_data_dir + "/zk.json"
        cls.simdiss_results = os.path.join(cls.note_data_dir, "simdiss.json")

    @classmethod
    def clean_notes(cls):

        try:
            with open(cls.note_data) as f:
                notes = json.load(f)
        except:
            print(f"Could not open note data at: {cls.note_data}")
            exit(1)

        # taking the note body, and not raw, means frontmatter is not included.
        dirty_notes = [note["body"] for note in notes]

        cleaned_notes = []

        for note in dirty_notes:
            note = re.sub(r'\n', ' ', note)
            note = re.sub(md_link_patt, r"\1", note)
            note = re.sub(md_symbols_patt, "", note)

            cleaned_notes.append(note)

        cls.cleaned_notes = cleaned_notes

    @classmethod
    def build_reference_data(cls):
        with open(cls.note_data) as f:
            notes = json.load(f)

        titles = [title["title"] for title in notes]
        paths = [path["path"] for path in notes]

        titles_dict = {}

        # for getting note index from note title
        l = len(titles)
        for i in range(0, l):
            titles_dict[titles[i]] = i

        cls.titles_dict = titles_dict
        cls.titles = titles
        cls.paths = paths

    @classmethod
    def generate_embeddings(cls, model):
        cls.corpus_embeddings = model.encode(cls.cleaned_notes)

    # Return embeddings of entire corpus
    @classmethod
    def embeddings(cls):
        return cls.corpus_embeddings

    # Lookup the index of a note by title
    @classmethod
    def get_index_from_title(cls, title):
        return cls.titles_dict[title]
