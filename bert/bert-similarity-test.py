# https://www.sbert.net/docs/quickstart.html

# check if correct conda env is active first
import os

env = os.environ['CONDA_PREFIX']
env = os.path.basename(env)

if env != "thesis":
    print("'thesis' conda env not active. Exiting program.")
    exit(1)

######## begin

import json, nltk

from sentence_transformers import SentenceTransformer
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()

nltk.download("punkt")

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("sentence-t5-base")

with open('data/ps.json') as f:
    notes = json.load(f)

note_bodies = [note["body"] for note in notes]
for note in note_bodies:
    detokenizer.detokenize(word_tokenize(note))
    print("----new note----")
    print(note)

# The sentences to encode
# note_bodies = [
#     "The weather is lovely today.",
#     "It's so sunny outside!",
#     "He drove to the stadium.",
# ]

# 2. Calculate embeddings by calling model.encode()
# embeddings = model.encode(note_bodies)
#
# # 3. Calculate the embedding similarities
# similarities = model.similarity(embeddings, embeddings)
#
# min, _ = torch.min(similarities, dim=0, keepdim=False)
# max, _ = torch.max(similarities, dim=0, keepdim=False)
# print(min, max)
# tensor([[1.0000, 0.6660, 0.1046], # first sentence
#         [0.6660, 1.0000, 0.1411], # second sentence
#         [0.1046, 0.1411, 1.0000]]) # third sentence

