# Comparing CRF and BI-LSTM networsk for Named Entity Recognition (NER)

Goal of this project is to compare the performance of conditional random fields (CRF) with a deep learning approach (Bidirectional Long-Short-Term-Memory network) for named entity recognition (NER).


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
* frequency of word in training data
* LDA topic (TODO)
* wordnet hypernym paths
* word is in nltk name/word list

## Results

See the current state of results in the [results](./results.md) file.

## Datasets:

Two datasets are used for training and evaluating the NER system:

* `WNUT 17 Emerging Entities`: contains text from Twitter, Stack Overflow responses, YouTube comments, and Reddit comments.

```
Leon Derczynski, Eric Nichols, Marieke van Erp, Nut Limsopatham; 2017. Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition. In Proceedings of the Workshop on Noisy, User-generated Text, at EMNLP. (https://noisy-text.github.io/2017/pdf/WNUT18.pdf)
```

* CoNLL 2003 datset from https://www.clips.uantwerpen.be/conll2003/ner/

## How-To

There are three files that provide a command line interface:

    * crf.py
    * lstm.py
    * lstm_crf.py

All of the three come with a very similar interface:

### CRF interace

```
"""crf, a tool to train/test the CRF NER system.

Usage:
  crf.py MODELNAME  [--rand=<samplesize>] [-f <file>]
  crf.py (-h | --help)

Options:
  -f --file             Input file with train/test data [defaults ./data/conll/eng.all].
  --rand=<samplesize>   Size of random sample [defaults: 5].
  -h --help             Show this screen.
"""
```

### BI-LSTM interface
```
lstm, a tool to train/test the LSTM NN for NER.

Usage:
  lstm.py MODELNAME [-f <file>] [--rand=<samplesize>] [--pretrain=<embeddings>]
  lstm.py (-h | --help)

Options:
  -f --file               Input file with train/test data [defaults ./data/conll/eng.all].
  --rand=<samplesize>     Pretty-print a sample of size samplesize [defaults: 5].
  --pretrain=<embeddings> File of pretrained embeddings. Per default embeddings are trained from scratch.
  -h --help               Show this screen.
```

### BI-LSTM-CRF interface
```
lstm_crf, a tool to train/test the LSTM NN for NER.

Usage:
  lstm_crf.py MODELNAME [-f <file>] [--rand=<samplesize>] [--pretrain=<embeddings>]
  lstm_crf.py (-h | --help)

Options:
  -f --file               Input file with train/test data [defaults ./data/conll/eng.all].
  --rand=<samplesize>     Pretty-print a sample of size samplesize [defaults: 5].
  --pretrain=<embeddings> File of pretrained embeddings. Per default embeddings are trained from scratch.
  -h --help               Show this screen.
```


## Sources:

The code (and the choosen features) are inspired by following blogs/papers/repos:

### CRF

* https://www.depends-on-the-definition.com/named-entity-recognition-conditional-random-fields-python/
* http://www.albertauyeung.com/post/python-sequence-labelling-with-crf/
* http://www.oegai.at/konvens2012/proceedings/17_tkachenko12o/17_tkachenko12o.pdf
* https://www.aclweb.org/anthology/D11-1141.pdf
* https://noisy-text.github.io/2016/pdf/WNUT22.pdf
* https://github.com/nguyeho7/CZ_NER
* https://github.com/Orbifold/dutch-ner/blob/master/NER%20using%20CRF.ipynb
* https://stackoverflow.com/questions/52049511/how-to-perform-clustering-on-word2vec
* https://github.com/aleju/ner-crf
* https://github.com/mayoyamasaki/py-kmeans-word2vec
* https://github.com/percyliang/brown-cluster
* https://stackabuse.com/python-for-nlp-topic-modeling/


### Bidirectional LSTM NN
* http://www.depends-on-the-definition.com/guide-sequence-tagging-neural-networks-python/
* https://github.com/Akshayc1/named-entity-recognition
* https://adventuresinmachinelearning.com/keras-lstm-tutorial/
* https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
* http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
* https://arxiv.org/pdf/1508.01991v1.pdf
