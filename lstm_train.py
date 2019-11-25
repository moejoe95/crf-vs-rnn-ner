#!/usr/bin/env python
"""lstm_train, a tool to train the CNN NER system with a given conll file.

Usage:
  lstm_train.py [(-m | --model) <model>] [(-t | --train) <file>]
  lstm_train.py (-h | --help)

Options:
  -h --help     Show this screen.
"""
from docopt import docopt
import conll_parser
import numpy as np
from keras.models import Sequential, Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

arguments = docopt(__doc__, version='lstm_train')

test_file = arguments.get('<file>')
test_file = './data/conll/eng.train' if test_file == None else test_file
model_name = arguments.get('<model>')
model_name = 'cnn.h5' if model_name == None else model_name

# parse file
docs, sentences, labels = conll_parser.parse(test_file)

# flatten 2D lists
words = [w for sen in sentences for w in sen]
labels = [l for sen in labels for l in sen]

# unique
words = list(set(words))
labels = list(set(labels))

# Dictionary word:index 
word2idx = {w : i for i, w in enumerate(words)}
labels2idx = {'I-LOC': 0, 'I-PER': 1, 'B-LOC': 2, 'O': 3, 'I-ORG': 4, 'B-MISC': 5, 'B-ORG': 6, 'I-MISC': 7}

max_len = 100 # hard coded max length of sentence

# words
X = [[word2idx[w[0]] for w in s] for s in docs]
X = pad_sequences(maxlen = max_len, sequences = X, padding = "post", value = 0)

# labels
y = [[labels2idx[w[1]] for w in s] for s in docs]
y = pad_sequences(maxlen = max_len, sequences = y, padding = "post", value = 0)
y = [to_categorical(i, num_classes = len(labels)) for i in y]

# Model architecture

hidden_size = 500
model = Sequential()
model.add(Embedding(len(words), hidden_size, input_length=max_len))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(len(labels))))
model.add(Activation('softmax'))


optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

checkpointer = ModelCheckpoint(filepath = model_name, verbose = 1, mode = 'auto', save_best_only = True, monitor='val_loss')

model.fit(X, np.array(y), batch_size=32, epochs=3, callbacks=[checkpointer])

model.save(model_name)