# https://www.sbert.net/docs/quickstart.html

# check if correct conda env is active first
import json
import re
import torch

import nltk
from nltk import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from sentence_transformers import SentenceTransformer


nltk.download("punkt")

model = SentenceTransformer("sentence-t5-base")
wd = TreebankWordDetokenizer()
wt = TreebankWordTokenizer()

md_symbols_patt = r"(#)|(-)|(>)|(\*)|(\")"
md_link_patt = r"\[(.*?)\]\(.*?\)"

## get an clean data

with open("data/ps.json") as f:
    notes = json.load(f)


note_bodies = [note["body"] for note in notes]

cleaned_notes = [""] * len(note_bodies)
dirty_notes = [""] * len(note_bodies)

for i, note in enumerate(note_bodies):
    note = note.replace("\n", " ")  # flatten for regex ease of use
    note = re.sub(md_link_patt, r"\1", note)
    note = re.sub(md_symbols_patt, "", note)
    cleaned_notes[i] = note
    # note_toks += sent_tokenize(note)
    # note_toks = dtkn.detokenize(note_toks)
    # note_body_toks += note_toks

# write to file
# with open("data/cleaned_notes.txt", "w") as cleaned_notes_file:
#     for n in cleaned_notes:
#         cleaned_notes_file.write(n)
#         cleaned_notes_file.write("\n\n")

for i, note in enumerate(note_bodies):
    note = note.replace("\n", " ")  # flatten for regex ease of use
    dirty_notes[i] = note


# Calculate embeddings
embeddings_clean = model.encode(cleaned_notes)
similarities_clean = model.similarity(embeddings_clean, embeddings_clean)
min_clean, _ = torch.min(similarities_clean, dim=0, keepdim=False)

embeddings_dirty = model.encode(dirty_notes)
similarities_dirty = model.similarity(embeddings_dirty, embeddings_dirty)
min_dirty, _ = torch.min(similarities_dirty, dim=0, keepdim=False)

print("min dirty\n")
print(min_dirty, "\n")
print("min clean\n")
print(min_clean, "\n")
print("dirty - clean\n")
diff = min_dirty - min_clean
print(diff)
