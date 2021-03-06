#!/usr/bin/env python
"""lstm_crf, a tool to train/test the LSTM NN for NER.

Usage:
  lstm_crf.py MODELNAME [-f <file>] [--rand=<samplesize>] [--pretrain=<embeddings>]
  lstm_crf.py (-h | --help)

Options:
  -f --file               Input file with train/test data [defaults ./data/conll/eng.all].
  --rand=<samplesize>     Pretty-print a sample of size samplesize [defaults: 5].
  --pretrain=<embeddings> File of pretrained embeddings. Per default embeddings are trained from scratch.
  -h --help               Show this screen.
"""
from docopt import docopt
import conll_parser
import numpy as np
import os
from keras.models import Model, Input, load_model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import Constants
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
import reports
import embeddings

arguments = docopt(__doc__, version='lstm_crf')

test_file = arguments.get('<file>')
if test_file == None:
  test_file = './data/conll/eng.all'
  print('no file specified, use default:', test_file, '...')

model_name = arguments.get('MODELNAME')
rand = arguments.get('--rand')

# parse file
docs, words, labels = conll_parser.parse(test_file)

if rand is not None:
  docs = conll_parser.filter_parsed(docs, rand)

# flatten 2D lists
words = [w for sen in words for w in sen]
words.append('-PAD-')

labels = [l for sen in labels for l in sen]
labels.append('-PAD-')

# unique
words = sorted(list(set(words))) # sorted is important here, otherwise the saved model can't be used again
labels = sorted(list(set(labels)))

# Dictionary word:index 
word2idx = {w : i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}

labels2idx = {w : i for i, w in enumerate(labels)}
idx2label = {i: w for w, i in labels2idx.items()}

# get longest sentence
max_len = len(max(words, key=len)) + 1

X = [[word2idx[w[0]] for w in s] for s in docs]
y = [[labels2idx[w[1]] for w in s] for s in docs]

# pad sentences to length of max_len
X = pad_sequences(maxlen = max_len, sequences = X, padding = "post", value = word2idx['-PAD-'])
y = pad_sequences(maxlen = max_len, sequences = y, padding = "post", value = labels2idx['-PAD-'])

# convert labels to categories
y = [to_categorical(i, num_classes = len(labels2idx)) for i in y]

# split data
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=Constants.TEST_SPLIT, random_state=Constants.RAND_SEED)

if rand is not None:
  X_te = X
  y_te = y


embedding_out_size = 256
dropout_rate = 0.2
dropout_rate_recurrent = 0.1
lstm_out_size = 50

input = Input(shape=(max_len,))
pretrain = arguments.get('--pretrain')
if pretrain is not None:
  print('use pre trained embeddings:', pretrain)
  model = embeddings.getPreTrainedEmbeddingLayer(word2idx, len(words), pretrain, 100, max_len)(input)
else:
  model = Embedding(input_dim=len(words), output_dim=embedding_out_size, input_length=max_len)(input)
model = Dropout(dropout_rate)(model)
model = Bidirectional(LSTM(units=lstm_out_size, return_sequences=True, recurrent_dropout=dropout_rate_recurrent))(model)
model = TimeDistributed(Dense(lstm_out_size, activation="relu"))(model)
crf = CRF(len(labels))  # CRF layer
out = crf(model)  # output

model = Model(input, out)

if not os.path.isfile(model_name):
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=20, validation_split=0.1, verbose=1)
    model.save(model_name)
else:
    custom_objects = {'CRF': CRF,
                    'crf_loss': crf_loss,
                    'crf_viterbi_accuracy': crf_viterbi_accuracy
                    }
    model = load_model(model_name, custom_objects=custom_objects)

# plot_model(model, to_file='lstm_crf.png')

# Evaluation
y_pred = model.predict(X_te)
y_pred = np.argmax(y_pred, axis=-1)
y_test_act = np.argmax(y_te, axis=-1)

y_pred = [[idx2label[i] for i in row] for row in y_pred]
y_test_act = [[idx2label[i] for i in row] for row in y_test_act]

# print report
if rand is None:
  reports.save_lstm_result(y_pred, y_test_act, X_te, idx2word, model_name + '.txt')
else:
  reports.rand_pretty_print(docs, y_pred)
  