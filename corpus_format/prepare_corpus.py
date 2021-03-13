import os
from utils.word2vec import DTACorpusPrepared
from utils.word2vec import NORMALIZED_CHARS_DTA
import nltk
# Change this to german if needed
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import re
import string
regex = re.compile('[%s]' % re.escape(string.punctuation))


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
    # Split the corpus into paragraphs and pass them to parse_paragraphs_nltk see e.g. prepareCOHA.py
    pass









