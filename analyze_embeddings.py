from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity as cosine_sim
import numpy as np
from matplotlib import pyplot as plt
import warnings
import copy
from gensim.models import Word2Vec
import re
from german.embedding_change.word2vec import EpochLogger
warnings.filterwarnings("error")


def merge_mapped_embeddings(emb1, emb2, modifier):
    """
    Merge the embeddings of emb2 into emb1. Modify the words in emb2 with 'modifier' to distinguish them from
    the words in emb1
    :param emb1: KeyedVectors instance
    :param emb2: KeyedVectors instance
    :return: merged KeyedVectors instance
    """
    merged_emb = copy.copy(emb1)
    word_list = [word + modifier for word in emb2.vocab]
    vec_list = [emb2[word] for word in emb2.vocab]
    merged_emb.add(word_list, vec_list)
    return merged_emb


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
    #words_vectors_dim_reduced = {word: pca_model.transform(words_vectors[word].reshape(1, -1)) for word in words_vectors}

    for i, dct in enumerate(res):
        print("Word:", word + modifiers[i])
        print(list(dct.keys()))
        print()
    #plot_words(words_vectors_dim_reduced)


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
            if embeddings_epoch1.vocab[word].count > 10000 and embeddings_epoch2.vocab[word].count > 10000:
                similarities[word] = (similarity[0], embeddings_epoch1.vocab[word].count, embeddings_epoch2.vocab[word].count)

    similarities_sorted = sorted(similarities, key=lambda word: similarities[word][0])
    return similarities_sorted


def is_in_all_vocabs(word, listofkv):
    for kv in listofkv:
        if word not in kv.vocab:
            return False
    return True


def syntactic_semantic_change(syntactic_embeddings_epoch1, syntactic_embeddings_epoch2, semantic_embeddings_epoch1, semantic_embeddings_epoch2, thresholds=[1.0, 0.1]):
    similarities_differences = []
    cosine_sim = syntactic_embeddings_epoch1.cosine_similarities
    for word in syntactic_embeddings_epoch1.vocab:
        if is_in_all_vocabs(word, [syntactic_embeddings_epoch1, syntactic_embeddings_epoch2, semantic_embeddings_epoch1, semantic_embeddings_epoch2]):
            if syntactic_embeddings_epoch1.vocab[word].count > MIN_COUNT and syntactic_embeddings_epoch2.vocab[word].count > MIN_COUNT:
                syntactic_similarity = cosine_sim(syntactic_embeddings_epoch1[word], syntactic_embeddings_epoch2[word].reshape(1, 100))[0]
                semantic_similarity = cosine_sim(semantic_embeddings_epoch1[word], semantic_embeddings_epoch2[word].reshape(1, 100))[0]
                if syntactic_similarity < thresholds[0] and semantic_similarity < thresholds[1]:
                    #similarities_differences[word] = (syntactic_similarity, embeddings_epoch1.vocab[word].count, embeddings_epoch2.vocab[word].count)
                    print(word)
                    print(syntactic_similarity)
                    print(semantic_similarity)
                    similarities_differences.append(word)

    #similarities_sorted = sorted(similarities, key=lambda word: similarities[word][0])
    return similarities_differences


def find_abitrary_words_with_similarity(syntactic_embeddings_epoch1, syntactic_embeddings_epoch2, semantic_embeddings_epoch1, semantic_embeddings_epoch2):
    similarities_differences_words = []
    cosine_sim = syntactic_embeddings_epoch1.cosine_similarities
    for word in syntactic_embeddings_epoch1.vocab:
        if len(similarities_differences_words) > 5:
            break
        if word in syntactic_embeddings_epoch2.vocab and word in semantic_embeddings_epoch1.vocab and word in semantic_embeddings_epoch2.vocab:
            if syntactic_embeddings_epoch1.vocab[word].count > MIN_COUNT and syntactic_embeddings_epoch2.vocab[word].count > MIN_COUNT and len(word) > 5:
                for other_word in syntactic_embeddings_epoch2.vocab:
                    if other_word in syntactic_embeddings_epoch1.vocab and other_word in semantic_embeddings_epoch1.vocab and other_word in semantic_embeddings_epoch2.vocab:
                        if syntactic_embeddings_epoch1.vocab[other_word].count > MIN_COUNT and syntactic_embeddings_epoch2.vocab[other_word].count > MIN_COUNT and len(other_word) > 5:
                            syntactic_similarity = \
                            cosine_sim(syntactic_embeddings_epoch1[word], syntactic_embeddings_epoch2[other_word].reshape(1, 100))[0]
                            semantic_similarity = \
                            cosine_sim(semantic_embeddings_epoch1[word], semantic_embeddings_epoch2[other_word].reshape(1, 100))[0]
                            if abs(syntactic_similarity) < 0.1 and semantic_similarity > 0.7:
                                similarities_differences_words.append((word, other_word))
                                if len(similarities_differences_words) > 2:
                                    break

    return similarities_differences_words


def find_abitrary_words_with_similarity(syntactic_embeddings_epoch1, syntactic_embeddings_epoch2, semantic_embeddings_epoch1, semantic_embeddings_epoch2):
    similarities_differences_words = []
    cosine_sim = syntactic_embeddings_epoch1.cosine_similarities
    for word in syntactic_embeddings_epoch1.vocab:
        if len(similarities_differences_words) > 5:
            break
        if is_in_all_vocabs(word, syntactic_embeddings_epoch1, syntactic_embeddings_epoch2, semantic_embeddings_epoch1, semantic_embeddings_epoch2):
            if syntactic_embeddings_epoch1.vocab[word].count > MIN_COUNT and syntactic_embeddings_epoch2.vocab[word].count > MIN_COUNT and len(word) > 5:
                syntactic_similarity = \
                    cosine_sim(syntactic_embeddings_epoch1[word],
                               syntactic_embeddings_epoch2.vectors)
                semantic_similarity = \
                    cosine_sim(semantic_embeddings_epoch1[word],
                               semantic_embeddings_epoch2.vectors)

                for other_word in syntactic_embeddings_epoch2.vocab:
                    if other_word in syntactic_embeddings_epoch1.vocab and other_word in semantic_embeddings_epoch1.vocab and other_word in semantic_embeddings_epoch2.vocab:
                        if syntactic_embeddings_epoch1.vocab[other_word].count > MIN_COUNT and syntactic_embeddings_epoch2.vocab[other_word].count > MIN_COUNT and len(other_word) > 5:
                            syntactic_similarity = \
                            cosine_sim(syntactic_embeddings_epoch1[word], syntactic_embeddings_epoch2[other_word].reshape(1, 100))[0]
                            semantic_similarity = \
                            cosine_sim(semantic_embeddings_epoch1[word], semantic_embeddings_epoch2[other_word].reshape(1, 100))[0]
                            if abs(syntactic_similarity) < 0.1 and semantic_similarity > 0.7:
                                similarities_differences_words.append((word, other_word))
                                if len(similarities_differences_words) > 2:
                                    break

    return similarities_differences_words


def find_abitrary_words_with_similarity(syntactic_embeddings_epoch1, syntactic_embeddings_epoch2, semantic_embeddings_epoch1, semantic_embeddings_epoch2):
    similarities_differences_words = []
    cosine_sim = syntactic_embeddings_epoch1.cosine_similarities
    for word in syntactic_embeddings_epoch1.vocab:
        if len(similarities_differences_words) > 5:
            break
        if is_in_all_vocabs(word, syntactic_embeddings_epoch1, syntactic_embeddings_epoch2, semantic_embeddings_epoch1, semantic_embeddings_epoch2):
            if syntactic_embeddings_epoch1.vocab[word].count > MIN_COUNT and syntactic_embeddings_epoch2.vocab[word].count > MIN_COUNT and len(word) > 5:
                syntactic_similarity = \
                    cosine_sim(syntactic_embeddings_epoch1[word],
                               syntactic_embeddings_epoch2.vectors)
                semantic_similarity = \
                    cosine_sim(semantic_embeddings_epoch1[word],
                               semantic_embeddings_epoch2.vectors)

            

                for other_word in syntactic_embeddings_epoch2.vocab:
                    if other_word in syntactic_embeddings_epoch1.vocab and other_word in semantic_embeddings_epoch1.vocab and other_word in semantic_embeddings_epoch2.vocab:
                        if syntactic_embeddings_epoch1.vocab[other_word].count > MIN_COUNT and syntactic_embeddings_epoch2.vocab[other_word].count > MIN_COUNT and len(other_word) > 5:
                            syntactic_similarity = \
                            cosine_sim(syntactic_embeddings_epoch1[word], syntactic_embeddings_epoch2[other_word].reshape(1, 100))[0]
                            semantic_similarity = \
                            cosine_sim(semantic_embeddings_epoch1[word], semantic_embeddings_epoch2[other_word].reshape(1, 100))[0]
                            if abs(syntactic_similarity) < 0.1 and semantic_similarity > 0.7:
                                similarities_differences_words.append((word, other_word))
                                if len(similarities_differences_words) > 2:
                                    break

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
    syntactic_embeddings1617 = KeyedVectors.load_word2vec_format(syn_path_epoch1617, binary=False)
    syn_path_epoch1819 = "/ukp-storage-1/deboer/Language-change/german/embedding_change/attract-repel-master/results/final_vectors_w2v.txt"
    syntactic_embeddings1819 = KeyedVectors.load_word2vec_format(syn_path_epoch1819, binary=False)

    # Load Semantic Embeddings
    sem_path_epoch1617 = "/ukp-storage-1/deboer/Language-change/german/embedding_change/1600-1700/word2vec/1617_emb_cleaned_v2_largedict_mapped.txt"
    semantic_embeddings1617 = KeyedVectors.load_word2vec_format(sem_path_epoch1617, binary=False)
    sem_path_epoch1819 = "/ukp-storage-1/deboer/Language-change/german/embedding_change/1800-1900/fasttext/1819_emb_cleaned_v3_cased.txt"
    semantic_embeddings1819 = KeyedVectors.load_word2vec_format(sem_path_epoch1819, binary=False)

    #load_word_counts(sem_path_epoch1617.replace('_largedict_mapped.txt', '.model'), syntactic_embeddings1617, semantic_embeddings1617)
    #load_word_counts(sem_path_epoch1819.replace('.txt', '.model'), syntactic_embeddings1819, semantic_embeddings1819)
    #print(semantic_embeddings1617.vocab[semantic_embeddings1617.index2word[0]].count, semantic_embeddings1617.index2word[0])
    #print(semantic_embeddings1617.vocab[semantic_embeddings1617.index2word[100]].count, semantic_embeddings1617.index2word[100])
    #print(semantic_embeddings1617.vocab[semantic_embeddings1617.index2word[5000]].count, semantic_embeddings1617.index2word[5000])
    #print(semantic_embeddings1617.vocab[semantic_embeddings1617.index2word[6000]].count, semantic_embeddings1617.index2word[6000])
    #print(semantic_embeddings1617.vocab[semantic_embeddings1617.index2word[5]].count, semantic_embeddings1617.index2word[5])

    # with open('/ukp-storage-1/deboer/Language-change/german/embedding_change/testout.txt', 'w') as f:
    #     for word in ['geht', 'drückt', 'griffen', 'lief', 'rief', 'ging', 'kennen', 'laufen', 'hält', 'kenne', 'Haus', 'Hauses', 'Häuser', 'rissen', 'eilten', 'Walde', 'Weges']:
    #         syn_similar = syntactic_embeddings1819.most_similar(positive=[word], topn=10)
    #         sem_similar = semantic_embeddings1819.most_similar(positive=[word], topn=10)
    #         f.write('###########################' + '\n')
    #         f.write(word + '\n')
    #         f.write(str(syn_similar) + '\n\n')
    #         f.write(str(sem_similar) + '\n')

    # count = semantic_embeddings1617.vocab['feurigsten'].count
    #
    # print(count)
    # pca_model = fit_pca(semantic_embeddings1617.vectors, semantic_embeddings1819.vectors)
    #
    #combined_embeddings = merge_mapped_embeddings(syntactic_embeddings1617, syntactic_embeddings1819, modifier=" (1800)")
    if RESULTS_SPACE == 'sem':
        combined_embeddings = merge_mapped_embeddings2([semantic_embeddings1617, semantic_embeddings1819], modifiers=[" (1600)", " (1800)"])
    elif RESULTS_SPACE == 'syn':
        combined_embeddings = merge_mapped_embeddings2([syntactic_embeddings1617, syntactic_embeddings1819], modifiers=[" (1600)", " (1800)"])
    changed_words = biggest_change_words()
    # # for word in ["sich", "ihr", "uns", "da", "wenn", "seine"]:
    # #    print(word)
    # #    plot_word_combined(word, combined_embeddings, pca_model, modifiers=["", " (1800)"])
    #
    # #changed_words = biggest_change_words(syntactic_embeddings1617, syntactic_embeddings1819)
    # print("Looking for relevant words...")
    # #changed_words = syntactic_semantic_change(syntactic_embeddings1617, syntactic_embeddings1819, semantic_embeddings1617, semantic_embeddings1819)
    # words_ab = find_abitrary_words_with_similarity(syntactic_embeddings1617, syntactic_embeddings1819, semantic_embeddings1617, semantic_embeddings1819)
    #
    # print("Plotting...")
    # for word_a, word_b in words_ab:
    #     print("word A: ", word_a)
    #     print("NNs", syn_combined_embeddings.most_similar(positive=[word_a + " (1600)"], topn=10))
    #     print("word B: ", word_b)
    #     print("NNs", syn_combined_embeddings.most_similar(positive=[word_b + " (1800)"], topn=10))
    #     print()
    # print('####SYN####')
    # for word in changed_words:
    #     if len(word) >= MIN_LENGTH:
    #         print(word)
    #         #plot_word(word, [semantic_embeddings1617, semantic_embeddings1819], pca_model, modifiers=[" (1600)", " (1800)"])
    #         plot_word_combined(word, syn_combined_embeddings, pca_model, modifiers=[" (1600)", " (1800)"])
    #
    # print('####SEM####')
    # for word in changed_words:
    #     if len(word) >= MIN_LENGTH:
    #         print(word)
    #         # plot_word(word, [semantic_embeddings1617, semantic_embeddings1819], pca_model, modifiers=[" (1600)", " (1800)"])
    #         plot_word_combined(word, sem_combined_embeddings, pca_model, modifiers=[" (1600)", " (1800)"])
