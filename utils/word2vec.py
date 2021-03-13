from gensim.utils import tokenize
import gensim.models.word2vec
from gensim.models.callbacks import CallbackAny2Vec
import os
from gensim.models.fasttext import FastText
import gensim

NORMALIZED_CHARS_DTA = {
    "oͤ": "ö",
    "ſ": "s",
    "uͤ": "ü",
    "ꝛ": "r",
    "aͤ": "ä",
    "- ": ""

}

NORMALIZED_CHARS_COHA = {

}


def normalize(string, normalization_dict):
    normalized_string = string
    for char, norm_char in normalization_dict.items():
        normalized_string = normalized_string.replace(char, norm_char)
    return normalized_string


def simple_preprocess(doc, deacc=False, min_len=2, max_len=15, lower=False):
    """Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long.

    Uses :func:`~gensim.utils.tokenize` internally.

    Parameters
    ----------
    doc : str
        Input document.
    deacc : bool, optional
        Remove accent marks from tokens using :func:`~gensim.utils.deaccent`?
    min_len : int, optional
        Minimum length of token (inclusive). Shorter tokens are discarded.
    max_len : int, optional
        Maximum length of token in result (inclusive). Longer tokens are discarded.
    lower : bool, optional
        Convert string to lowercase

    Returns
    -------
    list of str
        Tokens extracted from `doc`.

    """
    tokens = [
        token for token in tokenize(doc, lower=lower, deacc=deacc, errors='ignore')
        if min_len <= len(token) <= max_len and not token.startswith('_')
    ]
    return tokens


class DTACorpusPrepared(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, filepath, cleaning_dict=None):
        """
        :param filepath: path to a file with one sentence per line
        """
        self.corpus_path = filepath
        self.cleaning_dict = cleaning_dict

    def __iter__(self):
        for line in open(self.corpus_path):
            if self.cleaning_dict:
                normalized_line = normalize(line, self.cleaning_dict)
            else:
                normalized_line = line
            # assume there's one document per line, tokens separated by whitespace
            yield simple_preprocess(normalized_line)


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch: {}/{}".format(self.epoch + 1, model.epochs))

    def on_epoch_end(self, model):
        self.epoch += 1


def create_dict(gensim_model1, gensim_model2, size=100):
    """
    Write a pseudo dictionary between two different time epochs. Each word will be mapped to itself.
    The output format is word + whitespace + word as required by https://github.com/artetxem/vecmap.
    :param loaded gensim_model of epoch 1
    :param loaded gensim_model of epoch 2
    """
    most_freq1 = set(gensim_model1.wv.index2entity[:size])
    most_freq2 = set(gensim_model2.wv.index2entity[:size])

    common_words = []
    for word in most_freq1:
        if word in most_freq2:
            common_words.append(normalize(word))

    with open("/ukp-storage-1/deboer/freq_words_normalized3.dict", 'w') as f:
        for word in common_words:
            f.write(" ".join([word, word]))
            f.write('\n')


def merge_word2vec_files(file1, file2, modifier):
    with open(file1) as f1, open(file2) as f2:
        f1_lines = f1.readlines()
        f2_lines = f2.readlines()
    n_words1, dim1 = f1_lines[0].split(' ')
    n_words2, dim2 = f2_lines[0].split(' ')
    assert dim1 == dim2
    n_total = int(n_words1) + int(n_words2)
    i = 10
    k = 0
    with open("wv_all", "w") as all:
        all.write(" ".join(n_total, dim1))
        all.write("\n")
        all.writelines(f1_lines[1:i])
        for line in f2_lines:
            k += 1
            if k > i:
                break
            values = line.split(' ')
            # Modify the word of the second file to distinguish it from the first
            values[0] = values[0] + modifier
            all.write(" ".join(values))


def train_embeddings(corpusfile, save_model_to, method="word2vec", cleaning_dict=None):
    """
    Train and save embeddings for a corpus
    :param corpusfile: file with one sentence per line
    :param save_model_to: path to save the resulting model
    :param method: the method used to train the embeddings
    """
    # Make sure gensim is running with Cython
    assert gensim.models.word2vec.FAST_VERSION > -1

    epoch_logger = EpochLogger()
    sentences = DTACorpusPrepared(corpusfile, cleaning_dict=cleaning_dict)
    if method == 'word2vec':
        model = gensim.models.Word2Vec(sentences=sentences, callbacks=[epoch_logger])
    elif method == 'fasttext':
        model = gensim.models.FastText(sentences=sentences, callbacks=[epoch_logger])
    else:
        raise ValueError("method must be one of {word2vec, fasttext}")

    print("Training Finished.")
    # Save gensim model and word vectors in word2vec .txt format
    model.save(save_model_to + ".model")
    model.wv.save_word2vec_format(save_model_to + ".txt", binary=False)
    print("Model saved to", save_model_to)


if __name__ == '__main__':
    # Clean a corpus that is already in a format with one sentence per line
    corpus = DTACorpusPrepared("/ukp-storage-1/deboer/Language-change/german/embedding_change/1600-1700/1600-1700.txt",
                               cleaning_dict=NORMALIZED_CHARS_DTA)
    corpus_cleaned = "/ukp-storage-1/deboer/Language-change/german/embedding_change/1600-1700/1600-1700_cleaned_v3_cased.txt"
    if os.path.exists(corpus_cleaned):
        raise Exception("File already exists. Delete it if you want to overwrite")
    with open(corpus_cleaned, "w") as outfile:
        for pre_processed_line in corpus:
            outfile.write(" ".join(pre_processed_line))
            outfile.write("\n")

    # Train embeddings on cleaned corpus
    SAVE_MODEL_TO = "/ukp-storage-1/deboer/Language-change/german/embedding_change/1800-1900/fasttext/1819_emb_cleaned_v3_cased_dummy"

    train_embeddings(corpus_cleaned, SAVE_MODEL_TO, 'fasttext', cleaning_dict=NORMALIZED_CHARS_DTA)


    #create_dict(model1617, model1819, size=250)









