#!/usr/bin/env python
"""crf, a tool to train/test the CRF NER system on a given file.

Usage:
  crf.py MODELNAME  [--rand=<samplesize>] [-f <file>]
  crf.py (-h | --help)

Options:
  -f --file             Input file with train/test data.
  --rand=<samplesize>   Size of random sample [defaults: 5].
  -h --help             Show this screen.
"""

from docopt import docopt
import numpy as np
import pycrfsuite
import conll_parser
from FeatureGenerator import FeatureGenerator
import pos_tagger
import os
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split
import Constants
import reports
import random

arguments = docopt(__doc__, version='crf')

train_file = arguments.get('<file>')
train_file = './data/conll/eng.all' if train_file == None else train_file
model_name = arguments.get('MODELNAME')
rand = arguments.get('--rand')

# parse file
docs, words, _ = conll_parser.parse(train_file)

if rand is not None:
  docs = conll_parser.filter_parsed(docs, rand)

# do pos tagging, as part of feature extraction
data = pos_tagger.tag(docs)
feature = FeatureGenerator(data)
features = feature.extract_word_features()
labels = [feature.get_labels(doc) for doc in data]

X_tr, X_te, y_tr, y_te = train_test_split(features, labels, test_size=Constants.TEST_SPLIT, random_state=Constants.RAND_SEED)

if rand is not None:
  X_te = features
  y_te = labels

if not os.path.isfile(model_name):
  verb = True if Constants.VERBOSE == 1 else False
  trainer = pycrfsuite.Trainer(verbose=verb)

  # Submit training data to the trainer
  for xseq, yseq in zip(X_tr, y_tr):
      trainer.append(xseq, yseq)

  # Set the parameters of the model
  trainer.set_params({
      'c1': 0.5,
      'c2': 0.01,  
      'feature.possible_transitions': True
  })

  # save model to file
  trainer.train(model_name)

tagger = pycrfsuite.Tagger()
tagger.open(model_name)

y_pred = [tagger.tag(xseq) for xseq in X_te]

if rand is None:
  _, words, _, _ = train_test_split(words, labels, test_size=Constants.TEST_SPLIT, random_state=Constants.RAND_SEED)
  reports.save_to_file(y_pred, y_te, words, 'crf_conll.txt')
else:
  reports.rand_pretty_print(docs, y_pred)
