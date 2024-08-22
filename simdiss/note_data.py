import json
import regex as re


md_symbols_patt = r"(#)|(-)|(>)|(\*)|(\")"
md_link_patt = r"\[(.*?)\]\(.*?\)"


class NoteData:
    titles_arr = []
    titles_dict = {}
    cleaned_notes = []

    @classmethod
    def corpus(cls, path):
        with open(path) as f:
            notes = json.load(f)

        dirty_notes = [note["body"] for note in notes]
        titles_arr = [title["title"] for title in notes]

        titles_dict = {}
        for i, title in enumerate(titles_arr):
            titles_dict[title] = i

        cleaned_notes = [""] * len(dirty_notes)

        for i, note in enumerate(dirty_notes):
            note = note.replace("\n", " ")  # flatten for regex ease of use
            note = re.sub(md_link_patt, r"\1", note)
            note = re.sub(md_symbols_patt, "", note)
            cleaned_notes[i] = note

        cls.cleaned_notes = cleaned_notes
        cls.titles_arr = titles_arr
        cls.titles_dict = titles_dict

    @classmethod
    def note_bodies(cls):
        return cls.cleaned_notes

    @classmethod
    def titles(cls):
        return cls.titles_arr, cls.titles_dict

    @classmethod
    def index_from_title(cls, title):
        return cls.titles_dict[title]
