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

test_entities = 0
test_non_entities = 0
corr_pred_entities = 0
corr_pred_non_entities = 0
incorr_pred_entities = 0
incorr_pred_non_entities = 0

for i in range(0, len(y_pred)-1):
    for j in range(0, len(y_pred[i])-1):
        p = y_pred[i][j]
        t = test_labels[i][j]
        if t == 'O':
            test_non_entities += 1 
        else:
             test_entities += 1
        if p == t and p != 'O':
            corr_pred_entities += 1
        elif p == t and p == 'O':
            corr_pred_non_entities +=1
        elif p != t and p != 'O':
            incorr_pred_entities += 1
        elif p != t and p == 'O':
            incorr_pred_non_entities += 1

        else:
            print('test data invalid!')
            
print("number of words: ", test_entities + test_non_entities)
print("number of named entities: ", test_entities)
print("number of non named entities: ", test_non_entities)

print("correct predicted entities: ", corr_pred_entities)
print("correct predicted non entities: ", corr_pred_non_entities)
print("incorrect prediced entities: ", incorr_pred_entities)
print("incorrect predicted non entities: ", incorr_pred_non_entities)
