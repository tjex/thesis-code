from sentence_transformers import util
import torch

# Modified code of SentenceTransformers.util.cos_sim().
# This proves a [note x note] similarity matrix, which can be
# later queried against.
# The default implementation compairs similarity scores of two
# inputs: https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html


def cos_sim_elementwise(embeddings):
    a = util._convert_to_batch_tensor(embeddings)
    a_norm = util.normalize_embeddings(a)
    return torch.mm(a_norm, a_norm.transpose(0, 1))
