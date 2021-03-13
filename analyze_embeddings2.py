from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
import warnings
from gensim.models import Word2Vec
from utils.word2vec import EpochLogger
warnings.filterwarnings("error")


def fit_pca(*vectors):
    pca = PCA(n_components=2)
    pca.fit(np.vstack(vectors))
    return pca


def plot_word(word: str, embeddings: [KeyedVectors], pca_model: PCA, modifiers: [str]):
    words_vectors = {}
    for i, emb in enumerate(embeddings):
        words_vectors[word + modifiers[i]] = emb[word]
        similar = emb.most_similar(positive=[word], topn=5)
        for sim_word, _ in similar:
            words_vectors[sim_word + modifiers[i]] = emb[sim_word]

    # Apply dimensionality reduction before plotting
    words_vectors_dim_reduced = {word: pca_model.transform(words_vectors[word].reshape(1, -1)) for word in words_vectors}

    plot_words(words_vectors_dim_reduced)


def plot_words(words):
    x = [arr[0, 0] for arr in words.values()]
    y = [arr[0, 1] for arr in words.values()]
    labels = words.keys()
    colors = ['red'] * (len(words) // 2) + ['blue'] * (len(words) // 2)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(x, y, c = colors)

    for i, txt in enumerate(labels):
        ax.annotate(txt.replace('ſ', 's'), (x[i]+0.002, y[i]+0.002))

    try:
        plt.savefig("plotstest/" + list(labels)[0])
    except RuntimeWarning as e:
        print(e.with_traceback(e.__traceback__))
        print("Skipped one word")
    finally:
        plt.close()


def plot_word_combined(word: str, embedding: KeyedVectors, pca_model: PCA, modifiers: [str]):
    res = []
    for i, mod in enumerate(modifiers):
        words_vectors = {}
        words_vectors[word + mod] = embedding[word + mod]
        similar = embedding.most_similar(positive=[word + mod], topn=10)
        for sim_word, _ in similar:
            words_vectors[sim_word] = embedding[sim_word]
        res.append(words_vectors)

    # Apply dimensionality reduction before plotting
    # not in use anymore because it didn't help much when analyzing the outputs
    # Instead we just produce an output string
    #words_vectors_dim_reduced = {word: pca_model.transform(words_vectors[word].reshape(1, -1)) for word in words_vectors}
    res_string = ""
    for i, dct in enumerate(res):
        res_string += "Word:" + word + modifiers[i]
        res_string += str(list(dct.keys()))
        res_string += '\n'
    return res_string
    #plot_words(words_vectors_dim_reduced)


def merge_mapped_embeddings2(embs, modifiers):
    """
    Merge the embeddings into one KeyedVector instance. Modify the words of each provided embedding space with 'modifiers'
    to distinguish them from the each other i.e. modifiers must be a list with the same size embs
    :param embs: List of KeyedVectors instances to merge
    :param modifiers: modifiers for the words of each provided embedding space
    :return: merged KeyedVectors instance
    """
    merged_emb = KeyedVectors(100)
    for i, emb in enumerate(embs):
        word_list = [word + modifiers[i] for word in emb.vocab]
        vec_list = [emb[word] for word in emb.vocab]
        merged_emb.add(word_list, vec_list)
    return merged_emb


def filter_by_mincount(embeddings: KeyedVectors, min_count):
    """
    Eliminate all word that occur less than mincount times. Keep in mind that the counts of words are currently not
    copied to the returned KeyedVectors instance.
    """
    words = []
    vectors = []
    for word in embeddings.vocab:
        if embeddings.vocab[word].count > min_count:
            words.append(word)
            vectors.append(embeddings[word])
    filtered = KeyedVectors(embeddings.vector_size)
    filtered.add(words, vectors)
    return filtered


def biggest_change_words(embeddings_epoch1, embeddings_epoch2):
    similarities = {}
    cosine_sim = embeddings_epoch1.cosine_similarities
    for word in embeddings_epoch1.vocab:
        if word in embeddings_epoch2.vocab:
            similarity = cosine_sim(embeddings_epoch1[word], embeddings_epoch2[word].reshape(1, 100))
            similarities[word] = (similarity[0], embeddings_epoch1.vocab[word].count, embeddings_epoch2.vocab[word].count)

    similarities_sorted = sorted(similarities, key=lambda word: similarities[word][0])
    return similarities_sorted


def is_in_all_vocabs(word, listofkv):
    for kv in listofkv:
        if word not in kv.vocab:
            return False
    return True


def syntactic_semantic_change(syn_emb_ep1, syn_emb_ep2, sem_emb_ep1, sem_emb_ep2, syn_threshold, sem_threshold):
    similarities_differences = []
    cosine_sim = syn_emb_ep1.cosine_similarities
    for word in syn_emb_ep1.vocab:
        if is_in_all_vocabs(word, [syn_emb_ep1, syn_emb_ep2, sem_emb_ep1, sem_emb_ep2]):
            syntactic_similarity = cosine_sim(syn_emb_ep1[word], syn_emb_ep2[word].reshape(1, 100))[0]
            semantic_similarity = cosine_sim(sem_emb_ep1[word], sem_emb_ep2[word].reshape(1, 100))[0]
            if syntactic_similarity < syn_threshold and semantic_similarity > sem_threshold:
                print(word)
                print(syntactic_similarity)
                print(semantic_similarity)
                similarities_differences.append(word)
    return similarities_differences


def find_abitrary_words_with_similarity(syn_emb_ep1, syn_emb_ep2, sem_emb_ep1, sem_emb_ep2, syn_threshold, sem_threshold):
    similarities_differences_words = []
    cosine_sim = syn_emb_ep1.cosine_similarities
    for word in syn_emb_ep1.vocab:
        if len(similarities_differences_words) > 5:
            break
        if is_in_all_vocabs(word, [syn_emb_ep1, syn_emb_ep2, sem_emb_ep1, sem_emb_ep2]):
            syn_similarities = cosine_sim(syn_emb_ep1[word], syn_emb_ep2.vectors)
            # Get the indices where the similarities fulfill a certain condition
            indx = np.transpose(np.nonzero(syn_similarities >= syn_threshold))
            for id in indx:
                other_word = syn_emb_ep2.vocab[id]
                sem_similarity = cosine_sim(sem_emb_ep1[word], sem_emb_ep2[other_word].reshape(1, 100))[0]
                if abs(sem_similarity) > sem_threshold:
                    similarities_differences_words.append((word, other_word))

    return similarities_differences_words


def load_word_counts(gensim_model_path, syn_kv, sem_kv):
    w2v_model = Word2Vec.load(gensim_model_path)
    for word in w2v_model.wv.vocab:
        word_count = w2v_model.wv.vocab[word].count
        sem_kv.vocab[word].count = word_count
        syn_kv.vocab[word].count = word_count


if __name__ == '__main__':
    DO_PCA_PLOTS = False
    RESULTS_SPACE = 'syn'  # either 'syn' or 'sem'
    MIN_COUNT = 15
    MIN_LENGTH = 4

    # Load Syntactic Embeddings
    syn_path_epoch1617 = "/ukp-storage-1/deboer/Language-change/german/embedding_change/1600-1700/wang2vec/1617_emb_cleaned_v2_largedict_mapped.txt"
    syn_emb1617 = KeyedVectors.load_word2vec_format(syn_path_epoch1617, binary=False)
    syn_path_epoch1819 = "/ukp-storage-1/deboer/Language-change/german/embedding_change/attract-repel-master/results/final_vectors_w2v.txt"
    syn_emb1819 = KeyedVectors.load_word2vec_format(syn_path_epoch1819, binary=False)

    # Load Semantic Embeddings
    sem_path_epoch1617 = "/ukp-storage-1/deboer/Language-change/german/embedding_change/1600-1700/word2vec/1617_emb_cleaned_v2_largedict_mapped.txt"
    sem_emb1617 = KeyedVectors.load_word2vec_format(sem_path_epoch1617, binary=False)
    sem_path_epoch1819 = "/ukp-storage-1/deboer/Language-change/german/embedding_change/1800-1900/fasttext/1819_emb_cleaned_v3_cased.txt"
    sem_emb1819 = KeyedVectors.load_word2vec_format(sem_path_epoch1819, binary=False)

    load_word_counts(sem_path_epoch1617.replace('_largedict_mapped.txt', '.model'), syn_emb1617, sem_emb1617)
    load_word_counts(sem_path_epoch1819.replace('.txt', '.model'), syn_emb1819, sem_emb1819)

    syn_emb1617 = filter_by_mincount(syn_emb1617, MIN_COUNT)
    syn_emb1819 = filter_by_mincount(syn_emb1819, MIN_COUNT)
    sem_emb1617 = filter_by_mincount(sem_emb1617, MIN_COUNT)
    sem_emb1819 = filter_by_mincount(sem_emb1819, MIN_COUNT)


    if RESULTS_SPACE == 'sem':
        combined_embeddings = merge_mapped_embeddings2([sem_emb1617, sem_emb1819], modifiers=[" (1600)", " (1800)"])
    elif RESULTS_SPACE == 'syn':
        combined_embeddings = merge_mapped_embeddings2([syn_emb1617, syn_emb1819], modifiers=[" (1600)", " (1800)"])
    else:
        raise ValueError("RESULTS_SPACE must be either \'syn\' or \'sem\'")


    changed_words = biggest_change_words(syn_emb1617, syn_emb1819)

    changed_words_sym_sem = syntactic_semantic_change(syn_emb1617, syn_emb1819, sem_emb1617, sem_emb1819)

    with open('syn_sem_change.txt') as f:
        for word in changed_words:
            if len(word) >= MIN_LENGTH:
                f.write(word + '\n')
                f.write(plot_word_combined(word, combined_embeddings, None, modifiers=[" (1600)", " (1800)"]))


    words_ab = find_abitrary_words_with_similarity(syn_emb1617, syn_emb1819, sem_emb1617, sem_emb1819)

    with open("abr_syn_sem.txt", 'w') as f:
        for word_a, word_b in words_ab:
            f.write("word A: " + word_a + '\n')
            f.write("NNs" + str(combined_embeddings.most_similar(positive=[word_a + " (1600)"], topn=10)) + '\n')
            f.write("word B: " + word_b + '\n')
            f.write("NNs" + str(combined_embeddings.most_similar(positive=[word_b + " (1800)"], topn=10)) + '\n')
            f.write('\n')
