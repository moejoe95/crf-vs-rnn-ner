import numpy as np
import pycrfsuite
from sklearn.metrics import classification_report
import sys

import conll_parser
import pos_tagger
from FeatureGenerator import FeatureGenerator

if len(sys.argv) != 2:
    print("invalid number of arguments")
    exit(-1)

# parse file
docs = conll_parser.parse("test.conll")

# do pos tagging, as part of feature extraction
data = pos_tagger.tag(docs)
feature = FeatureGenerator()
features = feature.extract_word_features(data)

# split test / train data
_, X_test, _, y_test = feature.get_train_test_split(data, features)

# Generate predictions
tagger = pycrfsuite.Tagger()
tagger.open(sys.argv[1])
y_pred = [tagger.tag(xseq) for xseq in X_test]

correct = incorrect = 0
for i in range(0, len(y_pred)-1):
    for j in range(0, len(y_pred[i])-1):
        p = y_pred[i][j]
        t = y_test[i][j]
        if p == t:
            correct += 1
        else:
            incorrect +=1

# TODO calculate accuracy, recall, ...

print("number of correct classified named entities: ", correct)
print("number of incorrect classified named entities: ", incorrect)
