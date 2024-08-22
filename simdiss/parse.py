import json
import regex as re


md_symbols_patt = r"(#)|(-)|(>)|(\*)|(\")"
md_link_patt = r"\[(.*?)\]\(.*?\)"

def corpus(path):
    with open(path) as f:
        notes = json.load(f)


    dirty_notes = [note["body"] for note in notes]
    note_titles = [title["title"] for title in notes]

    cleaned_notes = [""] * len(dirty_notes)

    for i, note in enumerate(dirty_notes):
        note = note.replace("\n", " ")  # flatten for regex ease of use
        note = re.sub(md_link_patt, r"\1", note)
        note = re.sub(md_symbols_patt, "", note)
        cleaned_notes[i] = note

    return cleaned_notes, note_titles
