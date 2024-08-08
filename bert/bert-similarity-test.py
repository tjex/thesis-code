# https://www.sbert.net/docs/quickstart.html

# check if correct conda env is active first
import os

env = os.environ['CONDA_PREFIX']
env = os.path.basename(env)

if env != "thesis":
    print("'thesis' conda env not active. Exiting program.")
    exit(1)


from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("sentence-t5-base")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046], # first sentence
#         [0.6660, 1.0000, 0.1411], # second sentence
#         [0.1046, 0.1411, 1.0000]]) # third sentence

