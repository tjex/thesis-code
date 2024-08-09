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


# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(cleaned_notes)

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)

min, _ = torch.min(similarities, dim=0, keepdim=False)
# max, _ = torch.max(similarities, dim=0, keepdim=False)

print(min)

# tensor([[1.0000, 0.6660, 0.1046], # first sentence
#         [0.6660, 1.0000, 0.1411], # second sentence
#         [0.1046, 0.1411, 1.0000]]) # third sentence
