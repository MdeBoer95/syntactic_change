from demorphy import Analyzer
from demorphy.cache import memoize, lrudecorator
import pandas as pd
import itertools
from utils.syn_sem_pairs import remove_contradicting_pairs, write_word_pairs, read_stopwords, get_words_from_vocab
from utils.ordered_set import OrderedSet
import gensim
import os
from utils.word2vec import EpochLogger

DF_COLUMNS = ['CATEGORY', 'TENSE', 'NUMERUS', 'PERSON', 'MODE', 'DEGREE', 'CASE', 'GENDER']
CATEGORIES = ['V', 'ADJ', 'NN']


def build_word_pairs(words, max_pairs=None):
    word_infos = []
    for word in words:
        analyzer = Analyzer(char_subs_allowed=True)
        cache_size = 200  # you can arrange the size or unlimited cache. For German lang, we recommed 200 as cache size.
        cached = memoize if cache_size == "unlim" else (lrudecorator(cache_size) if cache_size else (lambda x: x))
        analyze = cached(analyzer.analyze)
        # The analyzer returns multiple possible forms
        # we take the last one because it often seems to be the most intuitive
        demorph_candidates = analyze(word)
        if len(demorph_candidates) > 0:
            candidate_attr = demorph_candidates[-1]._fields
            if candidate_attr['CATEGORY'] in CATEGORIES:
                # remove all information that we don't need i.e. everything but the attributes in DF_COLUMNS
                word_info = {key: value for key, value in candidate_attr.items() if key in DF_COLUMNS}
                word_info['WORD'] = word
                word_infos.append(word_info)

    df_words = pd.DataFrame(word_infos)
    df_words = df_words.fillna('unk')
    verb_groups = df_words.loc[df_words['CATEGORY'] == 'V'].groupby(['CATEGORY', 'TENSE', 'NUMERUS', 'PERSON', 'MODE'])
    verb_pairs = get_pairs_from_groupby(verb_groups)

    adj_groups = df_words.loc[df_words['CATEGORY'] == 'ADJ'].groupby(['CATEGORY', 'DEGREE', 'CASE', 'NUMERUS'])
    adj_pairs = get_pairs_from_groupby(adj_groups)

    noun_groups = df_words.loc[df_words['CATEGORY'] == 'NN'].groupby(['CATEGORY', 'CASE', 'GENDER', 'NUMERUS'])
    noun_pairs = get_pairs_from_groupby(noun_groups)
    return verb_pairs, adj_pairs, noun_pairs


def get_pairs_from_groupby(gb):
    word_pairs = []
    i= 0
    for group_key in list(gb.groups.keys())[1:]:
        if (num_features(group_key) < 3):
            continue
        words = gb.get_group(group_key)['WORD']
        if (i<10):
            print(group_key, '+++', 'size:', len(words))
            print(words)
            print()
        i+=1
        word_pairs.extend(itertools.combinations(words.tolist(), 2))
    conj_words = OrderedSet()
    for pair in word_pairs:
        conj_words.update(list(pair))
    pairs_cleaned = remove_contradicting_pairs(word_pairs, conj_words, verbose=False)
    return pairs_cleaned


def num_features(groupkey):
    fts = [ft for ft in groupkey if ft != 'unk']
    return len(fts)


if __name__ == '__main__':
    MIN_COUNT = 100
    MODEL_PATH = "/home/marcel/Schreibtisch/gensim_sample/1819_emb_cleaned_v3_cased.model"  # gensim embedding model
    STOPWORDS_PATH = "data/stopwords-de.txt"
    PAIR_SEPARATOR = " "
    OUTPUT_DIR = "/home/marcel/Schreibtisch"

    # Load gensim model for vocab
    model = gensim.models.Word2Vec.load(MODEL_PATH)
    stopwords = read_stopwords(STOPWORDS_PATH)

    words = get_words_from_vocab(gensim_model=model, min_count=MIN_COUNT, stopwords=stopwords, max_words=2000)
    verb_pairs, adj_pairs, noun_pairs = build_word_pairs(words)

    write_word_pairs(verb_pairs, os.path.join(OUTPUT_DIR, 'verb_pairs'), sep=PAIR_SEPARATOR)
    write_word_pairs(adj_pairs, os.path.join(OUTPUT_DIR, 'adj_pairs'), sep=PAIR_SEPARATOR)
    write_word_pairs(noun_pairs, os.path.join(OUTPUT_DIR, 'noun_pairs'), sep=PAIR_SEPARATOR)


