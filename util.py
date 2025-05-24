import os, torch
import numpy as np
from numpy.typing import NDArray

embeddings_file = "data/embeddings.npy"
similarities_file = "data/similarities.pt"

def save_embeddings(embeddings):
    print(f"Saving embeddings to: {embeddings_file} ...")
    np.save(embeddings_file, embeddings)

def load_embeddings() -> NDArray:
    if not os.path.isfile(embeddings_file):
        print(f"Could not find {embeddings_file}. Run: sm train")
        exit(1)

    embeddings = np.load(embeddings_file)
    return embeddings


def save_similarities(similarities):
    print(f"Saving similarities to: {similarities_file} ...")
    torch.save(similarities, similarities_file)

def load_similarities():
    if not os.path.isfile(similarities_file):
        print(f"Could not find {similarities_file}. Run: sm train")
        return

    print("Loading similarities...")
    similarities = torch.load("data/similarities.pt")
    return similarities

