#!/usr/bin/env python
"""cnn_test, a tool to test the CNN NER system with a given conll file.

Usage:
  cnn_test.py [(-m | --model) <model>] [(-t | --train) <file>]
  cnn_test.py (-h | --help)

Options:
  -h --help     Show this screen.
"""
import conll_parser
import numpy as np
from docopt import docopt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from sklearn_crfsuite.metrics import flat_classification_report

arguments = docopt(__doc__, version='cnn_test')

test_file = arguments.get('<file>')
test_file = './data/conll/eng.testa' if test_file == None else test_file
model_name = arguments.get('<model>')
model_name = 'cnn.h5' if model_name == None else model_name

# parse file
docs, sentences, labels = conll_parser.parse(test_file)

# flatten 2D lists
words = [w for sen in sentences for w in sen]
labels = [l for sen in labels for l in sen]

words = list(set(words))
labels = list(set(labels))

word2idx = {w : i for i, w in enumerate(words)}
labels2idx = {'I-LOC': 0, 'I-PER': 1, 'B-LOC': 2, 'O': 3, 'I-ORG': 4, 'B-MISC': 5, 'B-ORG': 6, 'I-MISC': 7}
idx2label = {i: w for w, i in labels2idx.items()}

max_len = 100

# words
X = [[word2idx[w[0]] for w in s] for s in docs]
X = pad_sequences(maxlen = max_len, sequences = X, padding = "post", value = 0)

# labels
y = [[labels2idx[w[1]] for w in s] for s in docs]
y = pad_sequences(maxlen = max_len, sequences = y, padding = "post", value = 0)
y = [to_categorical(i, num_classes = len(labels2idx.items())) for i in y]

model = load_model(model_name)

# Evaluation
y_pred = model.predict(X)
y_pred = np.argmax(y_pred, axis=-1)
y_test_true = np.argmax(y, axis=-1)

# Convert the index to tag
y_pred = [[idx2label[i] for i in row] for row in y_pred]
y_test_true = [[idx2label[i] for i in row] for row in y_test_true]

report = flat_classification_report(y_pred=y_pred, y_true=y_test_true)
print(report)