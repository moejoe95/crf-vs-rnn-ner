#!/usr/bin/env python
"""crf_test, a tool to test the CRF NER system on a given model on a given file.

Usage:
  crf_test.py [(-m | --model) <model>] [(-t | --test) <file>]
  crf_test.py (-h | --help)

Options:
  -h --help     Show this screen.
"""

from docopt import docopt
import numpy as np
import pycrfsuite
from sklearn.metrics import classification_report
import sys

import conll_parser
import pos_tagger
import test_validation
from FeatureGenerator import FeatureGenerator
from sklearn_crfsuite.metrics import flat_classification_report

arguments = docopt(__doc__, version='crf_test')

test_file = arguments.get('<file>')
test_file = './data/conll/eng.testa' if test_file == None else test_file
model_name = arguments.get('<model>')
model_name = 'crf.model' if model_name == None else model_name


# parse file
docs, words, labels = conll_parser.parse(test_file)

# do pos tagging, as part of feature extraction
data = pos_tagger.tag(docs)
feature = FeatureGenerator(data)
features = feature.extract_word_features()
test_labels = [feature.get_labels(doc) for doc in data]

# Generate predictions
tagger = pycrfsuite.Tagger()
tagger.open(model_name)
y_pred = [tagger.tag(xseq) for xseq in features]

report = flat_classification_report(y_pred=y_pred, y_true=test_labels)
print(report)
