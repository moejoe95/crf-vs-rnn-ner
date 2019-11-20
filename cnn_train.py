#!/usr/bin/env python
"""cnn_train, a tool to train the CNN NER system with a given conll file.

Usage:
  cnn_train.py (-m | --model) <model> (-t | --train) <file>
  cnn_train.py (-h | --help)

Options:
  -h --help     Show this screen.
"""
from docopt import docopt
import conll_parser
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
import numpy as np
from sklearn.model_selection import train_test_split
from cnn_preprocessing import get_padded_seq


arguments = docopt(__doc__, version='crf_test')

test_file = arguments.get('<file>', './data/conll/eng.testa')
model_name = arguments.get('<model>', 'cnn.h5')

# parse file
docs = conll_parser.parse(test_file)

X, y, label_count, word_count, max_len = get_padded_seq(docs)
print(label_count)

input = Input(shape=(max_len,))
model = Embedding(input_dim=word_count, output_dim=50, input_length=max_len)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(label_count, activation="softmax"))(model)  # softmax output layer

model = Model(input, out)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X, np.array(y), batch_size=16, epochs=1, verbose=1)

model.save(model_name)
