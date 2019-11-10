# Comparing CRF and NN for Named Entity Recognition (NER)

Goal of this project is to compare the performance of conditional random fields (CRF) with a deep learning approach like RNNs for named entity recognition (NER).

## Current state

At the moment this project just contains an implementation using CRF.

## Sources:

The code (and the choosen features) are inspired by following blogs/papers:

* https://www.depends-on-the-definition.com/named-entity-recognition-conditional-random-fields-python/
* http://www.albertauyeung.com/post/python-sequence-labelling-with-crf/
* http://www.oegai.at/konvens2012/proceedings/17_tkachenko12o/17_tkachenko12o.pdf
* http://xpo6.com/list-of-english-stop-words/
* https://www.aclweb.org/anthology/D11-1141.pdf
* https://noisy-text.github.io/2016/pdf/WNUT22.pdf
* https://github.com/nguyeho7/CZ_NER
* https://github.com/Orbifold/dutch-ner/blob/master/NER%20using%20CRF.ipynb
* https://stackoverflow.com/questions/52049511/how-to-perform-clustering-on-word2vec

## Dataset:

The `WNUT 17 Emerging Entities` dataset is used for training and testing the classifier. It contains text from Twitter, Stack Overflow responses, YouTube comments, and Reddit comments.

Leon Derczynski, Eric Nichols, Marieke van Erp, Nut Limsopatham; 2017. Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition. In Proceedings of the Workshop on Noisy, User-generated Text, at EMNLP. (https://noisy-text.github.io/2017/pdf/WNUT18.pdf)

