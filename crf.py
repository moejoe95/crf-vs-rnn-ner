#!/usr/bin/env python
"""crf, a tool to train/test the CRF NER system with a given conll file.

Usage:
  crf.py (-m | --model) <model> [(-t | --train) <file>] 
  crf.py (-h | --help)

Options:
  -h --help     Show this screen.
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

arguments = docopt(__doc__, version='crf')

train_file = arguments.get('<file>')
train_file = './data/conll/eng.all' if train_file == None else train_file
model_name = arguments.get('<model>')

# parse file
docs, words = conll_parser.parse(train_file)

# do pos tagging, as part of feature extraction
data = pos_tagger.tag(docs)
feature = FeatureGenerator(data)
features = feature.extract_word_features()
labels = [feature.get_labels(doc) for doc in data]

X_tr, X_te, y_tr, y_te = train_test_split(features, labels, test_size=0.15)

if not os.path.isfile(model_name):
  trainer = pycrfsuite.Trainer(verbose=True)

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

report = flat_classification_report(y_pred=y_pred, y_true=y_te)
print(report)
