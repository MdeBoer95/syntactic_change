import json
import os
import argparse
from utils.similarity_measures import cosine_sim, dot_product_sim


def read_parsed_corpus(filepath):
    with open(filepath) as f:
        syn_features = json.load(f)
    return syn_features


def is_word_interesting(features_dict_c1, features_dict_c2, min_occurrence_diff=10):
    for featstring, counts in features_dict_c1.items():
        if featstring in features_dict_c2:
            if abs(counts - features_dict_c2[featstring]) > min_occurrence_diff:
                return True
        else:
            if counts > min_occurrence_diff:
                return True
    return False


def total_occurrences(features_dict):
    return sum(features_dict.values())


def print_word_features(word, features_dict_c1, features_dict_c2):
    print(word_features_pretty_string(word, features_dict_c1, features_dict_c2))
    print()


def word_features_pretty_string(word, features_dict_c1, features_dict_c2):
    pretty_string = ""
    pretty_string += "###Word: " + word + '\n'
    pretty_string += "##Features corpus 1:\n"
    for featstring, count in features_dict_c1.items():
        pretty_string += featstring + ': ' + str(count) + '\n'
    pretty_string += "##Features corpus 2:\n"
    for featstring, count in features_dict_c2.items():
        pretty_string += featstring + ': ' + str(count) + '\n'
    pretty_string += '\n'
    return pretty_string


def find_word_class_distribution(parsed_c1, parsed_c2, min_occurrences, word_classes, min_sim, sim_measure):
    """
    Return a dict of words with an interesting word distribution. In this context interesting means that there has
    been a change from corpus 1 to corpus 2
    """
    word_class_distributions = {}
    for word_class in parsed_c1:
        if total_occurrences(parsed_c1[word_class]) < min_occurrences:
            continue
        word, _ = word_class.split('_')
        class_counts_c1 = {wclass: 0 for wclass in word_classes}
        class_counts_c2 = {wclass: 0 for wclass in word_classes}
        for wclass in word_classes:
            word_wclass = '_'.join([word, wclass])
            if word_wclass in parsed_c1 and word_wclass in parsed_c2:
                class_counts_c1[wclass] += total_occurrences(parsed_c1[word_wclass])
                class_counts_c2[wclass] += total_occurrences(parsed_c2[word_wclass])
                if is_word_class_distribution_interesting(class_counts_c1, class_counts_c2, min_sim,
                                                          measure=sim_measure):
                    word_class_distributions[word] = [class_counts_c1, class_counts_c2]
    i = 0
    return word_class_distributions


def is_word_class_distribution_interesting(distribution_dict_c1, distribution_dict_c2, min_sim=0.5, measure=cosine_sim):
    normalized_c1 = normalize_count_dict(distribution_dict_c1)
    normalized_c2 = normalize_count_dict(distribution_dict_c2)

    sim = measure(list(normalized_c1.values()), list(normalized_c2.values()))
    return sim < min_sim


def normalize_count_dict(a_dict):
    total_count = sum(a_dict.values())
    return {key: value / total_count for key, value in a_dict.items()}


def find_interesting_words(parsed_c1, parsed_c2, min_occurrences, min_occurrences_diff):
    interesting_words = {}
    for word, features_dict_c1 in parsed_c1.items():
        if total_occurrences(features_dict_c1) > min_occurrences and word in parsed_c2:
            features_dict_c2 = parsed_c2[word]
            if is_word_interesting(features_dict_c1, features_dict_c2, min_occurrence_diff=min_occurrences_diff):
                interesting_words[word] = {'corpus1': features_dict_c1, 'corpus2': features_dict_c2}
    return interesting_words


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parsed_corpus1', required=True, type=str,
                        help='Path to the parsed corpus of the first epoch.')
    parser.add_argument('--parsed_corpus2', required=True, type=str,
                        help='Path to the parsed corpus of the second epoch.')
    parser.add_argument('--outdir', required=True, type=str, help='The path to the output directory.')
    parser.add_argument('--word_classes', required=False, default=['NOUN', 'ADJ', 'VERB', 'AUX'], nargs='+', type=str,
                        help='Word classes that will be considered. Make sure to name them in the same way as the trankit parser does.')
    parser.add_argument('--min_occurrences', required=False, default=200, type=int,
                        help='Minimum number of occurrences of a word with a specific pos tag to be considered interesting.')
    parser.add_argument('--min_occurrences_diff', required=False, default=200, type=int,
                        help='Minimum difference of the occurrences of a word the in a specific feature combination between the two corpora.')
    parser.add_argument('--min_sim', required=False, default=0.5, type=float,
                        help='Minimum Similarity between two vectors of word class distributions to be considered interesting.')
    parser.add_argument('--sim_measure', required=False, default='cosine', choices=['cosine', 'dotproduct'], type=str,
                        help='Similarity measure for comparing two vectors of word class distributions.')

    args = parser.parse_args()

    sim_functions = {'cosine': cosine_sim, 'dotproduct': dot_product_sim}
    sim_measure = sim_functions[args.sim_measure]

    word_feature_statistiks_c1 = read_parsed_corpus(args.parsed_corpus1)
    word_feature_statistiks_c2 = read_parsed_corpus(args.parsed_corpus2)

    interesting_words = find_interesting_words(word_feature_statistiks_c1, word_feature_statistiks_c2,
                                               args.min_occurrences, args.min_occurrences_diff)
    with open(os.path.join(args.outdir, 'feature_diff.json'), 'w') as f:
        json.dump(interesting_words, f)

    words_with_interesting_dist = find_word_class_distribution(word_feature_statistiks_c1, word_feature_statistiks_c2,
                                                               args.min_occurrences,
                                                               args.word_classes, args.min_sim, sim_measure)
    with open(os.path.join(args.outdir, 'wordclass_dist.json'), 'w') as f:
        json.dump(words_with_interesting_dist, f)
