from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def cosine_sim(v1, v2):
    cosine_sim = cosine_similarity(np.array(v1).reshape(1, -1), np.array(v2).reshape(1, -1))
    return cosine_sim


def dot_product_sim(v1, v2):
    dp_sim = np.dot(np.array(v1).reshape(1, -1), np.array(v2).reshape(-1, 1))
    return dp_sim