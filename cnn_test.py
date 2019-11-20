#!/usr/bin/env python
"""cnn_test, a tool to test the CNN NER system on a given model on a given file.

Usage:
  cnn_test.py (-m | --model) <model> (-t | --test) <file>
  cnn_test.py (-h | --help)

Options:
  -h --help     Show this screen.
"""

from docopt import docopt
import sys
import conll_parser
from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
from cnn_preprocessing import get_padded_seq
import test_validation

arguments = docopt(__doc__, version='crf_test')

test_file = arguments.get('<file>', './data/conll/eng.testa')
model = arguments.get('<model>', 'cnn.h5')

# parse file
docs = conll_parser.parse(test_file)

labels = []
words = []
for sentence in docs:
    for word in sentence:
        words.append(word[0])
        labels.append(word[1])

model = load_model(model)

X, y, _, _, _ = get_padded_seq(docs, shape=model.output_shape)

p = model.predict(np.array([X[0]]))
