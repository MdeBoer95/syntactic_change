import spacy
from german.embedding_change.prepare_corpus import parse_paragraphs_nltk
from zipfile import ZipFile
from tqdm import tqdm
import os
import logging

logging.info()
def load_sentencizer():
    """
    Create minimal spacy pipeline for rule based sentencizing
    :return:
    """
    nlp = spacy.blank('de', disable=["tagger", "parser", "ner"])
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    nlp.max_length = 10000000
    return nlp


if __name__ == '__main__':
    nlp = load_sentencizer()
    OUT_DIR = '/ukp-storage-1/deboer/Language-change/german/embedding_change/COHA_gensim_preprocessed'
    COHA_BASE_DIR = '/ukp-storage-1/deboer/Language-change/COHA'
    zipnames = ["text_1810s_kso.zip",
                "text_1820s_jsi.zip",
                "text_1830s_bsu.zip",
                "text_1840s_nsq.zip",
                "text_1850s_jpr.zip",
                "text_1860s_psi.zip",
                "text_1870s_vuy.zip",
                "text_1880s_azz.zip",
                "text_1890s_jsq.zip",
                "text_1900s_mse.zip",
                "text_1910s_jue.zip",
                "text_1920s_gte.zip",
                "text_1930s_bie.zip",
                "text_1940s_bsw.zip",
                "text_1950s_ndz.zip",
                "text_1960s_xus.zip",
                "text_1970s_qkn.zip",
                "text_1980s_bkk.zip",
                "text_1990s_bsj.zip",
                "text_2000s_scc.zip"]


    token_counts = {}
    for zipname in zipnames:
        with ZipFile(os.path.join(COHA_BASE_DIR, zipname), 'r') as azip:
            zip_date = int(zipname.split('_')[1][:-1])
            zip_bucket = zip_date // 50 * 50
            logging.info(msg="Parsing" + zipname)
            for name in azip.namelist():
                with azip.open(name, 'r') as txtfile:
                    lines = txtfile.readlines()
                    # the third line contains all the text
                    if len(lines) == 1 and lines[0].startswith(b'##'):
                        text = lines[0][2:]
                    elif len(lines) == 3:
                        text = lines[2]
                    else:
                        print("Unexpected Format in file", name, ". Skipping this file.")
                        continue
                    num_tokens = parse_paragraphs_nltk([text.decode('utf-8')], nlp_pipeline=nlp, bucket_name=zip_bucket, basedir=OUT_DIR)
                    if zip_bucket in token_counts:
                        token_counts[zip_bucket] += num_tokens
                    else:
                        token_counts[zip_bucket] = num_tokens
    logging.info(msg=str(token_counts))

