from trankit import Pipeline
import json
from tqdm import tqdm
import argparse


class TokenPosKey:
    """
    Class for storing a token+pos tag as a key in a dictionary
    """
    def __init__(self, token: str, pos: str):
        self.token = token
        self.pos = pos

    def __eq__(self, other):
        if isinstance(other, TokenPosKey):
            return self.token == other.token and self.pos == other.pos
        return False

    def __ne__(self, other):
        return not self.__eq__(self, other)

    def __hash__(self):
        return hash((self.token, self.pos))

    def __repr__(self):
        return self.token + "_" + self.pos

    def __str__(self):
        return self.__repr__()


def dict2featstring(feats_dict: dict):
    key_and_feature = [key + '=' + feats_dict[key] for key in sorted(feats_dict)]
    feats_string = '|'.join(key_and_feature)
    return feats_string


def featstring2dict(feats_string: dict):
    feats_dict = {}
    feats = feats_string.split('|')
    for feat in feats:
        feature, value = feat.split('=')
        feats_dict[feature] = value
    return feats_dict


def sort_featstring(featstring):
    return dict2featstring(featstring2dict(featstring))


def create_morph_feature_dict(trankit_pipeline, corpus_text, word_classes):
    all_tokens = {}
    for line in tqdm(corpus_text):
        if len(line) < 3:
            continue
        parse_output = trankit_pipeline.posdep(line)
        for sentence in parse_output['sentences']:
            for token in sentence['tokens']:
                if 'upos' not in token:
                    continue
                word_class = token['upos']
                if word_class not in word_classes or 'feats' not in token:
                    continue
                word = token['text']
                word_pos_key = str(TokenPosKey(word, word_class))  # group by word + po
                                                                   # need string here if we want to dump to json
                feats_string = token['feats']
                feats_string = sort_featstring(feats_string)
                if word_pos_key in all_tokens and feats_string in all_tokens[word_pos_key]:
                    all_tokens[word_pos_key][feats_string] += 1
                elif word_pos_key in all_tokens:
                    all_tokens[word_pos_key][feats_string] = 1
                else:
                    all_tokens[word_pos_key] = {feats_string: 1}
    return all_tokens


def read_corpus(corpus_path, chunk_size=100, pretokenize_on_space=False):
    all_chunks = []
    with open(corpus_path) as f:
        chunk_lines = []
        for i, line in enumerate(f):
            chunk_lines.append(line.strip())
            if len(chunk_lines) == chunk_size:
                if pretokenize_on_space:
                    chunk = [chunk_line.split(' ') for chunk_line in chunk_lines]
                else:
                    chunk = ". ".join(chunk_lines)
                all_chunks.append(chunk)
                chunk_lines = []
        # Process last remaining chunk if not already processed
        if len(chunk_lines) != chunk_size:
            if pretokenize_on_space:
                chunk = [chunk_line.split(' ') for chunk_line in chunk_lines]
            else:
                chunk = ". ".join(chunk_lines)
            all_chunks.append(chunk)
    return all_chunks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', required=True, type=str,
                        help='Path to the corpus to analyze. Format is expected to be one sentence per line.')
    parser.add_argument('--outpath', required=True, type=str, help='The path to the output file.')
    parser.add_argument('--pretokenize_on_space', required=False, default=False,
                        help='If True the sentences will be tokenized on space and passed to the parser. This can speed up computation.')
    parser.add_argument('--chunk_size', required=False, default=100,
                        help='Pass chunk_size sentences as one paragraph to the parser. Important for speeding up the computation.')
    parser.add_argument('--lang', required=False, default='german',
                        help='Language of the corpus. Will be passed to trankit parser.')
    parser.add_argument('--word_classes', required=False, default=['NOUN', 'ADJ', 'VERB', 'AUX'],
                        help='Word classes that will be considered. Make sure to name them in the same way as the trankit parser does')

    args = parser.parse_args()

    corpus_text = read_corpus(args.corpus_path, chunk_size=args.chunk_size,
                              pretokenize_on_space=args.pretokenize_on_space)
    pipeline = Pipeline(args.lang)

    parsed_corpus_dict = create_morph_feature_dict(pipeline, corpus_text, word_classes=args.word_classes)

    with open(args.outpath, 'w') as outfile:
        json.dump(parsed_corpus_dict, outfile)
