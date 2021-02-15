#from german import load_dta
#from german.parse import load_stanza
from german.parse_spacy import load_spacy
#from gensim import utils
#from gensim.test.utils import datapath
#import gensim
#import torch
import os
import csv
from german.load_dta import get_dta_buckets
from german.embedding_change.word2vec import DTACorpusPrepared
from german.embedding_change.word2vec import NORMALIZED_CHARS_DTA
from german.load_ll2 import get_ll2_buckets
from german.load_historical import get_historical_buckets
from nltk.tokenize import sent_tokenize
import nltk
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import re
import string
regex = re.compile('[%s]' % re.escape(string.punctuation))


def count_tokens(filepath):
    num_tokens = 0
    with open(filepath) as f:
        for line in f:
            num_tokens += len(utils.simple_preprocess(line))
    return num_tokens


def parse_paragraphs(paragraphs: [str], nlp_pipeline, bucket_name, basedir):
    num_tokens = 0
    filename = os.path.join(basedir, str(bucket_name))
    if os.path.exists(filename):
        #print("File", bucket_name, "already exists. Appending to file.")
        pass
    with open(filename, 'a') as outputfile:
        for i, paragraph in enumerate(paragraphs):
            if i % 10 == 0:
                #print("{}/{} done.".format(i, len(paragraphs)))
                pass

            paragraph = str(paragraph)#.replace('ſ','s')
            if len(paragraph) < 10:
                continue

            result = nlp_pipeline(paragraph)
            spacy_sentences = result.sents
            for spacy_sentence in spacy_sentences:

                # TODO: sanity checks go here
                if len(spacy_sentence) < 4:
                    continue

                outputfile.write(spacy_sentence.text)
                outputfile.write('\n')
                num_tokens += len(spacy_sentence)

                #torch.cuda.empty_cache()

    return num_tokens


def parse_paragraphs_nltk(paragraphs: [str], nlp_pipeline, bucket_name, basedir):
    num_tokens = 0
    filename = os.path.join(basedir, str(bucket_name))
    if os.path.exists(filename):
        #print("File", bucket_name, "already exists. Appending to file.")
        pass
    with open(filename, 'a') as outputfile:
        for i, paragraph in enumerate(paragraphs):
            if i % 10 == 0:
                #print("{}/{} done.".format(i, len(paragraphs)))
                pass

            paragraph = str(paragraph)#.replace('ſ','s')
            if len(paragraph) < 10:
                continue

            result = sent_detector.tokenize(paragraph.strip())
            for spacy_sentence in result:

                # TODO: sanity checks go here
                if len(regex.sub('', spacy_sentence).split()) < 3:
                    continue

                outputfile.write(spacy_sentence)
                outputfile.write('\n')
                num_tokens += len(spacy_sentence.split())

                #torch.cuda.empty_cache()

    return num_tokens


def combine2files(file1, file2, outfile="combined.txt"):
    with open(file1) as first, open(file2) as second, open(outfile, "w") as out:
        lines_first = first.readlines()
        lines_second = second.readlines()

        out.writelines(lines_first)
        out.writelines(lines_second)


if __name__ == '__main__':
    #BASEDIR = "/ukp-storage-1/deboer/Language-change/german/embedding_change/v3_tmp"
    #if not os.path.exists(BASEDIR):
    #    os.mkdir(BASEDIR)

    #spacy_pipeline = load_spacy()
    #buckets = get_dta_buckets()
    #token_counts = {}

    #for bucket_name, bucket_paragraphs in buckets.items():
    #    num_tokens = parse_paragraphs_nltk(bucket_paragraphs, None, bucket_name, basedir=BASEDIR)
    #    token_counts[bucket_name] = num_tokens
    #with open("counts.txt", "w") as c:
    #    c.write(str(token_counts))

    #filepath = "/home/marcel/Schreibtisch/1600-1700_preprocessed.txt"
    #num_tokens = count_tokens(filepath)
    #print("{} tokens in file {}".format(num_tokens, filepath))
    corpus = DTACorpusPrepared("/ukp-storage-1/deboer/Language-change/german/embedding_change/1600-1700/1600-1700.txt", cleaning_dict=NORMALIZED_CHARS_DTA)
    corpus_cleaned = "/ukp-storage-1/deboer/Language-change/german/embedding_change/1600-1700/1600-1700_cleaned_v3_cased.txt"
    if os.path.exists(corpus_cleaned):
        raise Exception("File already exists. Delete it if you want to overwrite")
    with open(corpus_cleaned, "w") as outfile:
        for pre_processed_line in corpus:
            outfile.write(" ".join(pre_processed_line))
            outfile.write("\n")

    # with open("/home/deboer/language_change/Language-change/german/embedding_change/1600-1700/1600-1700.txt") as a, open("/home/deboer/language_change/Language-change/german/embedding_change/wangprepro1617.txt") as b:
    #     a_lines = a.readlines()
    #     b_lines = b.readlines()
    #     print(a_lines[1])
    #     print(b_lines[1])
    #
    #     print(a_lines[1000])
    #     print(b_lines[1000])
    #
    #     print(a_lines[40000])
    #     print(b_lines[40000])
    #
    #     print(a_lines[-1])
    #     print(b_lines[-1])








