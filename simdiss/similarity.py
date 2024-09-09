from sentence_transformers import util
import torch


# modified code of SentenceTransformers.util.cos_sim().
# Adapted for singular use case of elementwise similarity.
def cos_sim_elementwise(embeddings):
    a = util._convert_to_batch_tensor(embeddings)
    a_norm = util.normalize_embeddings(a)
    return torch.mm(a_norm, a_norm.transpose(0, 1))
