import sys
import conll_parser
from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
from cnn_preprocessing import get_padded_seq


if len(sys.argv) != 2:
    print("invalid number of arguments")
    exit(-1)

# parse file
docs = conll_parser.parse("dataset.conll")

model = load_model(sys.argv[1])

X, y, _, _ = get_padded_seq(docs)

_, X_te, _, y_te = train_test_split(X, y, test_size=0.1)

p = model.predict(np.array(X_te))
print(p)
