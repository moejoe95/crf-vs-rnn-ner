import conll_parser
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
import numpy as np
from sklearn.model_selection import train_test_split
from cnn_preprocessing import get_padded_seq

# parse file
docs = conll_parser.parse("dataset.conll")

X, y, label_count, word_count, max_len = get_padded_seq(docs)

input = Input(shape=(max_len,))
model = Embedding(input_dim=word_count, output_dim=50, input_length=max_len)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(label_count, activation="softmax"))(model)  # softmax output layer

model = Model(input, out)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.9)
history = model.fit(X_tr, np.array(y_tr), batch_size=16, epochs=3, verbose=1)

model.save('cnn.h5')
