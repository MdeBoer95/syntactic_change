# Syntactic Language Change
Analyze the change in syntax of a language over time.

## Setup
TODO

## Parser Based Approach
This approach uses a linguistic parser (in our case [trankit](https://github.com/nlp-uoregon/trankit)) to determine syntactic change between two corpora. First the corpus is parsed
and morphological features are extracted for each word (*per default for Nouns, Adjectives, Verbs and Auxiliary Verbs see parameter **--word_classes** in the scripts*). 
Then the results of the parser are analyzed according to different criteria with the purpose of finding syntactic change between two corpora.



### Parsing the Corpus
In order to determine morphological features of words in the corpus a morphological parser ([trankit](https://github.com/nlp-uoregon/trankit)) is used. The resulting combinations of morphological features are then grouped and 
counted per word + pos-tag (e.g. Schwimmen_NN, schwimmen_V). As an example output you can see
the content of ```parsed_corpus/morph_features_1617.json``` and ```parsed_corpus/morph_features_1617.json```
(Which are the results for the dta corpus divided into texts from 1600-1799 and 1800-1920).
  
For this part the script ```word_features_parse.py``` is used.  
For example:
```commandline
python word_features_parse.py --corpus_path /path/to/my/preprocessed/corpus --outpath myparsed_corpus.json
```
**The script expects the corpus to be in a format with one sentence per line.**  
There are several optional parameters. You can execute the script with the **--help** flag to get more information.

### Analyzing the Parsed Results
For this part the script ```word_features_analyze.py``` is used.  
For example:
```commandline
python word_features_analyze.py --parsed_corpus1 /path/to/my/parsed/corpus_1.json --parsed_corpus2 /path/to/my/parsed/corpus_2.json --outpath syn_changed_words.json
```
**The script expects the output of the previous step as parsed corpus**  
As an example output you can check the content of ```word_features_results/results_cosinesim.json```.  


There are several optional parameters. You can execute the script with the **--help** flag to get more information.


Here the results of the parser are analyzed for syntactic change between to two corpora. The script considers words as *intersting* (meaning that their syntactical features might have changed) 
If one of the two following cases happen:

#### Distribution of Word Classes changed
This means that the relative frequencies of the considered word classes has changed MIN_DIFF according to a given distance measure.   
  
For example for the word **fliehen**: 

| Corpus        | word class distribution |
| ------------- |:-------------:|
| dta 1600-1799  |{"NOUN": 14, "ADJ": 3, "VERB": 710, "AUX": 0} | 
| dta 1800-1920   | {"NOUN": 1, "ADJ": 4, "VERB": 213, "AUX": 0}     |   


#### Combinations of features for a word occur significantly more often in corpus A than in corpus B
TODO: Write description

## Word Embedding Based Approach
TODO: Write description