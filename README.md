# Syntactic Language Change
Analyze the change in syntax of a language over time.

## Setup
Install the requirements
```commandline
pip install -r requirements.txt
```
If you want to you create word pairs, install [DEMorphy](https://github.com/DuyguA/DEMorphy).

## Data & Preprocessing
- For analysing syntactical change only the dta corpus has been used so far. Data and embeddings can be found 
  in ```/ukp-storage-1/deboer/Language-change/german/embedding_change``` in folders ```1600-1700``` and ```1800-1900```.
  Latest versions of the data are suffixed with v3 and v3_cased. The cleaning involves gensims *simple_preprocess* function 
  and some additional normalizations of special characters. See ```/utils/word2vec```.
  The file also contains code to train fasttext and word2vec embeddings. 
- Parlamentsdebatten does not contain enough tokens to train meaningful word embeddings if split into two corpora/epochs
- The COHA data has been preprocessed but not analyzed yet. Preprocessed Data can be found in ```/ukp-storage-1/deboer/Language-change/german/embedding_change/COHA_gensim_preprocessed```

Preprocessing the data also means that the format of the corpus will be rewritten into a format with one sentence per line before cleaning can be applied.
The relevant files for this are in the folder ```corpus_format```.

## Parser Based Approach
This approach uses a linguistic parser (in our case [trankit](https://github.com/nlp-uoregon/trankit)) to determine syntactic change between two corpora. First the corpus is parsed
and morphological features are extracted for each word (*per default for Nouns, Adjectives, Verbs and Auxiliary Verbs see parameter **--word_classes** in the scripts*). 
Then the results of the parser are analyzed according to different criteria with the purpose of finding syntactic change between two corpora.

### Parsing the Corpus
In order to determine morphological features of words in the corpus, a morphological parser ([trankit](https://github.com/nlp-uoregon/trankit)) is used. The resulting combinations of morphological features are then grouped and 
counted per word + pos-tag (e.g. Schwimmen_NN, schwimmen_V). 

As an example output you can see
the content of ```parsed_corpus/morph_features_1617.json``` and ```parsed_corpus/morph_features_1819.json```
(Which are the results for the *dta* corpus divided into texts from 1600-1799 and 1800-1920).
  
For this part the script ```word_features_parse.py``` is used.  
For example:
```commandline
python word_features_parse.py --corpus_path /path/to/my/preprocessed/corpus --outpath myparsed_corpus.json
```
***The script expects the corpus to be in a format with one sentence per line.***  
There are several optional parameters. You can execute the script with the **--help** flag to get more information.

### Analyzing the Parsed Results
For this part the script ```word_features_analyze.py``` is used.  
For example:
```commandline
python word_features_analyze.py --parsed_corpus1 /path/to/my/parsed/corpus_1.json --parsed_corpus2 /path/to/my/parsed/corpus_2.json --outdir syn_changed_words.json
```
***The script expects the output of the previous step as parsed corpus***  

There are several optional parameters. You can execute the script with the **--help** flag to get more information.


Here the results of the parser are analyzed for syntactic change between the two corpora. The script considers words as *interesting* (meaning that their syntactical features might have changed) 
if one of the two following cases happen:

#### 1. Distribution of Word Classes changed
This means that the relative frequencies of the considered word classes has changed **--min_sim** according to a given distance measure.   
For an example output check ```word_features_results/results_cosinesim.json```.  
For example for the word **fliehen**: 

| Corpus        | word class distribution |
| ------------- |:-------------:|
| dta 1600-1799  |{"NOUN": 14, "ADJ": 3, "VERB": 710, "AUX": 0} | 
| dta 1800-1920   | {"NOUN": 1, "ADJ": 4, "VERB": 213, "AUX": 0}     |   


#### 2. Combinations of features for a word occur significantly more often in corpus A than in corpus B
If a certain combination of features for a word e.g. ```Case=Dat|Gender=Fem|Number=Sing``` occurs significantly more/less often in corpus A than
in corpus B. (See parameter **--min_occurrences_diff**)  
For an example output check ```word_features_results/feature_diff.json```.  
The *total occurrences are not taken into account right now* i.e. the counts are not normalized. It probably makes sense
to normalize the counts by the total occurrences of a word in the respective corpus


## Word Embedding Based Approach
In order to determine syntactic change with an embedding space we first have to create a syntactic embedding space.
For this we use [attract-repel](https://github.com/nmrksic/attract-repel). 
Attract-repel in its original form uses pairs of antonyms and synonyms to improve semantic embedding spaces by pushing antonyms further apart 
and pulling synonyms closer together. Since we want to create a syntactic embedding space we want to use semantically 
similar words as antonyms and syntactically similar words as synonyms.

The directory ```external/``` contains a slightly modified version of the original attract repel code, that allows to use
cased embeddings and write the resulting embeddings in word2vec format.

For generating pairs of semantically/syntactically similar words, the scripts ```sem_pairs.py```/```syn_pairs.py```
can be used.

Once the syntactic embedding spaces have been created, the next step is to map the embeddings from two different epochs
into one common embedding space. In order to do this we use [vecmap](https://github.com/artetxem/vecmap) in ```semi-supervised```
mode.
The same mapping can be done for a pair of semantic embedding spaces.
The dictionary can be created with the function *create_dict* in ```utils/word2vec```

In order to analyze the embedding spaces the script ```analyze_embeddings.py```.
The global parameters are defined at the start of the main method and can be changed as required. 
(Paths to embeddings, min count, etc.)
The parameter *RESULTS_SPACE* whether the outputs will be in semantic or in syntatic space.

The script outputs 3 files:

(examples can be found in the folder ```analyze_embeddings_outputs_expl```)

1. The n syntactically most changed words between two epochs.The output file excludes words that are shorter than MIN_LENGTH
   n defaults to 50. (file: biggest_change.txt)
   The output shows the word and its nearest neighbours in the *RESULT_SPACE* in both epochs (1600 = 1600-1700, 1800=1800-1900)
2. Words the have changed semantically less than a certain threshold and syntactically more than a certain threshold. 
   the threshold can be set in the code (see function *syntactic_semantic_change*). (file: syn_sem_change.txt)
   The output has the same format as in **1.**
3. Two arbitrary words that fulfill the following criteria: (file: abr_syn_sem.txt)
     - semantic similarity > sem_threshold
     - |syntactic similariy| < syn_threshold
     - word a is from epoch 1 and word b is from epoch 2
    
    This function (*find_abitrary_words_with_similarity*) can take very long to run. So it might make break the for loop after a
    certain number of words has been found. (Currently ~ 20)
       
    
## Test for Syntactic Word Embeddings
A test for syntactic word embeddings can be run with the file ```analogy_test.txt```.
For example:
```commandline
python analogy_test.py /storage/nllg/compute-share/deboer/melvin/language_change/Language-change/german/embedding_change/1600-1700/word2vec/1617_emb_cleaned_mapped.txt 
```
The test also produces an output file named *analogy_test.txt* to show which tests failed. Delete or rename the file is you run the test multiple
times, otherwise the test will append more output to the same file.