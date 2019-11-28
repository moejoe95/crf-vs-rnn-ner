#!/usr/bin/env python
"""lstm_train, a tool to train and/or test the CNN NER system with a given conll file.

Usage:
  lstm_train.py [(-t | --train) <file>]
  lstm_train.py (-h | --help)

Options:
  -h --help     Show this screen.
"""
from docopt import docopt
import conll_parser
import numpy as np
from keras.models import Model, Input, load_model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split


arguments = docopt(__doc__, version='lstm_train')

test_file = arguments.get('<file>')
if test_file == None:
  test_file = './data/conll/eng.all'
  print('no file specified, use default:', test_file, '...')


# parse file
docs, sentences = conll_parser.parse(test_file)

# flatten 2D lists
words = [w for sen in sentences for w in sen]
words.append('-PAD-')

# unique
words = list(set(words))

# Dictionary word:index 
word2idx = {w : i for i, w in enumerate(words)}

labels2idx = labels = {
    'I-LOC':    0, 
    'B-LOC':    1, 
    'I-PER':    2, 
    'B-PER':    3,
    'I-ORG':    4,
    'B-ORG':    5, 
    'I-MISC':   6,
    'B-MISC':   7, 
    'O':        8,
    '-PAD-':    9
}
idx2label = {i: w for w, i in labels2idx.items()}

max_len = 100 # hard coded max length of sentence

# words
X = [[word2idx[w[0]] for w in s] for s in docs]
X = pad_sequences(maxlen = max_len, sequences = X, padding = "post", value = word2idx['-PAD-'])

# labels
y = [[labels2idx[w[1]] for w in s] for s in docs]
y = pad_sequences(maxlen = max_len, sequences = y, padding = "post", value = labels2idx['-PAD-'])
y = [to_categorical(i, num_classes = len(labels2idx)) for i in y]

# split data
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15)

# Model architecture
input = Input(shape=(max_len,))
model = Embedding(input_dim=len(words), output_dim=50, input_length=max_len)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(len(labels2idx), activation="softmax"))(model)  # softmax output layer
model = Model(input, out)

# compile and fit model
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5, validation_split=0.1, verbose=1)

# Evaluation
y_pred = model.predict(X_te)
y_pred = np.argmax(y_pred, axis=-1)
y_test_true = np.argmax(y_te, axis=-1)

y_pred = [[idx2label[i] for i in row] for row in y_pred]
y_test_true = [[idx2label[i] for i in row] for row in y_test_true]

# print report
report = flat_classification_report(y_pred=y_pred, y_true=y_test_true)
print(report)