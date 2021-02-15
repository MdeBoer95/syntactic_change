from utils.ordered_set import OrderedSet
from tqdm import tqdm
from pattern.de import lemma


def remove_contradicting_pairs(all_pairs, conjugated_words, disable_tqdm=False, verbose=True):
    conj_words = {word: OrderedSet() for word in conjugated_words}
    # Group all tuples by their first word
    for pair in all_pairs:
        word1 = pair[0]
        conj_words[word1].add(pair)

    # Make sure each conjugated word is only paired with one form of another word
    for word, pairs in tqdm(conj_words.items(), disable=disable_tqdm):
        pairs_cleaned = OrderedSet()
        paired_words = OrderedSet()
        for pair in pairs:
            word2 = pair[1]
            word2_lemma = lemma(word2)
            if word2_lemma not in paired_words:
                paired_words.add(word2_lemma)
                pairs_cleaned.add(pair)
            else:
                # If this word stem already appears here it must be removed. Otherwise syntactically different forms
                # of the same stem will appear with the same other word and hence be mapped closely together
                # in attract-repel
                if verbose:
                    print("Removed:", pair, "From:", pairs)
        conj_words[word] = pairs_cleaned

    all_pairs_cleaned = OrderedSet()
    for pairs in conj_words.values():
        all_pairs_cleaned.update(pairs)
    return all_pairs_cleaned


def write_word_pairs(pairs, filepath, sep='\t'):
    with open(filepath, 'w') as f:
        for pair in pairs:
            f.write(sep.join(pair))
            f.write('\n')


def read_stopwords(filepath):
    stopwords = OrderedSet()
    with open(filepath) as f:
        for line in f:
            stopword = line.strip()
            stopwords.add(stopword)
    return stopwords


def get_words_from_vocab(gensim_model, min_count, stopwords, max_words=None):
    words = []
    for word in gensim_model.wv.vocab:
        if max_words and len(words) >= max_words:
            break
        if gensim_model.wv.vocab[word].count > min_count and word not in stopwords:
            words.append(word)
    return words