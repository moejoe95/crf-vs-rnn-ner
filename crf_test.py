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
test_labels = [feature.get_labels(doc) for doc in data]

# Generate predictions
tagger = pycrfsuite.Tagger()
tagger.open(sys.argv[1])
y_pred = [tagger.tag(xseq) for xseq in features]

tp = tn = fp = fn = 0

for i in range(0, len(y_pred)-1):
    for j in range(0, len(y_pred[i])-1):
        p = y_pred[i][j]
        t = test_labels[i][j]

        if p != 'O' and t != 'O':
            tp += 1
        elif p == 'O' and t == 'O':
            tn +=1
        elif p != 'O' and t == 'O':
            fp += 1
        elif p == 'O' and t != 'O':
            fn += 1 
        else:
            print('test data invalid!')

accuracy = (tp+tn)/(tp+tn+fp+fn) 
precision = (tp) / (tp+fp) 
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

print("accuracy = ", accuracy)
print("precision = ", precision)
print("recall = ", recall)
print("f1 = ", f1)