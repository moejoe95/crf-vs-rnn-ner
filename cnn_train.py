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
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn_crfsuite.metrics import flat_classification_report
from keras.callbacks import ModelCheckpoint


arguments = docopt(__doc__, version='crf_test')

test_file = arguments.get('<file>', './data/conll/eng.testa')
model_name = arguments.get('<model>', 'cnn.h5')

# parse file
docs = conll_parser.parse(test_file)

words = []
labels = []
for sentence in docs:
    for word in sentence:
        words.append(word[0])
        labels.append(word[1])

words = list(set(words))
labels = list(set(labels))

# Dictionary word:index pair
# word is key and its value is corresponding index
word_to_index = {w : i + 2 for i, w in enumerate(words)}
word_to_index["UNK"] = 1
word_to_index["PAD"] = 0

labels_to_index = {t : i + 1 for i, t in enumerate(labels)}
labels_to_index["PAD"] = 0

idx2word = {i: w for w, i in word_to_index.items()}
idx2label = {i: w for w, i in labels_to_index.items()}

max_len = 100
num_labels = len(list(set(labels)))

# words
X = [[word_to_index[w[0]] for w in s] for s in docs]
X = pad_sequences(maxlen = max_len, sequences = X, padding = "post", value = word_to_index["PAD"])

# labels
y = [[labels_to_index[w[1]] for w in s] for s in docs]
y = pad_sequences(maxlen = max_len, sequences = y, padding = "post", value = labels_to_index["PAD"])
y = [to_categorical(i, num_classes = num_labels + 1) for i in y]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Model architecture
input = Input(shape=(max_len,))
model = Embedding(input_dim=len(words)+2, output_dim=50, input_length=max_len)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(num_labels+1, activation="softmax"))(model)

model = Model(input, out)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
checkpointer = ModelCheckpoint(filepath = 'model.h5', verbose = 0, mode = 'auto', save_best_only = True, monitor='val_loss')

# fit model
model.fit(X_train, np.array(y_train), batch_size=32, epochs=5, validation_split=0.2, callbacks=[checkpointer])

# Evaluation
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)
y_test_true = np.argmax(y_test, axis=-1)

# Convert the index to tag
y_pred = [[idx2label[i] for i in row] for row in y_pred]
y_test_true = [[idx2label[i] for i in row] for row in y_test_true]

report = flat_classification_report(y_pred=y_pred, y_true=y_test_true)
print(report)
