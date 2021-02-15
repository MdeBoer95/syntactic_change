from utils.syn_sem_pairs import get_words_from_vocab, read_stopwords, write_word_pairs
import gensim
import os


def build_word_pairs(words, keyed_vectors: gensim.models.KeyedVectors):
    word_pairs = []
    for word in words:
        sim_words = keyed_vectors.most_similar(positive=word, topn=3)
        pairs = [(word, sim_word[0]) for sim_word in sim_words]
        word_pairs.extend(pairs)
    return word_pairs


if __name__ == '__main__':
    MIN_COUNT = 100
    MODEL_PATH = "/ukp-storage-1/deboer/Language-change/german/embedding_change/1800-1900/fasttext/1819_emb_cleaned_v3_cased.model"
    #MODEL_PATH = "/home/marcel/Work/Hiwi_aiphes/Language-change/german/embedding_change/1600-1700/word2vec.model"
    STOPWORDS_PATH = "/ukp-storage-1/deboer/Language-change/german/embedding_change/DEMorphy/stopwords-de.txt"
    #STOPWORDS_PATH = "stopwords-de.txt"
    PAIR_SEPARATOR = " "
    OUTPUT_DIR = "/ukp-storage-1/deboer/Language-change/german/embedding_change/syn_pairs_data"

    model = gensim.models.Word2Vec.load(MODEL_PATH)
    stopwords = read_stopwords(STOPWORDS_PATH)

    words = get_words_from_vocab(gensim_model=model, min_count=MIN_COUNT, stopwords=stopwords, max_words=10000)
    sem_pairs = build_word_pairs(words, model.wv)

    write_word_pairs(sem_pairs, os.path.join(OUTPUT_DIR, 'sem_pairs'), sep=PAIR_SEPARATOR)