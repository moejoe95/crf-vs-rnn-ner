# Comparing CRF and NN for Named Entity Recognition (NER)

Goal of this project is to compare the performance of conditional random fields (CRF) with a deep learning approach like RNNs (or LTSMs) for named entity recognition (NER).

## Current state

At the moment this project just contains an implementation using CRF.

## Features used for CRF

* word length
* 3 char prefix of word
* 3 char suffix of word
* word contains upper case char
* word contains non alphanumeric char
* if word is a stopword
* the word shape (Foobar has shape Aa+, BAR has shape A+)
* word2vec cluster using kmeans clustering
* brown cluster & bitsequence
* gazetter (build up from training data) lookup (TODO)
* frequency of word in training data
* LDA topic (TODO)


## Sources:

The code (and the choosen features) are inspired by following blogs/papers/repos:

### CRF

* https://www.depends-on-the-definition.com/named-entity-recognition-conditional-random-fields-python/
* http://www.albertauyeung.com/post/python-sequence-labelling-with-crf/
* http://www.oegai.at/konvens2012/proceedings/17_tkachenko12o/17_tkachenko12o.pdf
* http://xpo6.com/list-of-english-stop-words/
* https://www.aclweb.org/anthology/D11-1141.pdf
* https://noisy-text.github.io/2016/pdf/WNUT22.pdf
* https://github.com/nguyeho7/CZ_NER
* https://github.com/Orbifold/dutch-ner/blob/master/NER%20using%20CRF.ipynb
* https://stackoverflow.com/questions/52049511/how-to-perform-clustering-on-word2vec
* https://github.com/aleju/ner-crf
* https://github.com/mayoyamasaki/py-kmeans-word2vec
* https://github.com/percyliang/brown-cluster
* https://stackabuse.com/python-for-nlp-topic-modeling/


### LSTM
* http://www.depends-on-the-definition.com/guide-sequence-tagging-neural-networks-python/
* https://github.com/Akshayc1/named-entity-recognition



## Datasets:

Two datasets are used for evaluating the NER system:

* `WNUT 17 Emerging Entities`: contains text from Twitter, Stack Overflow responses, YouTube comments, and Reddit comments.

```
Leon Derczynski, Eric Nichols, Marieke van Erp, Nut Limsopatham; 2017. Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition. In Proceedings of the Workshop on Noisy, User-generated Text, at EMNLP. (https://noisy-text.github.io/2017/pdf/WNUT18.pdf)
```

* CoNLL 2003 datset from https://www.clips.uantwerpen.be/conll2003/ner/

