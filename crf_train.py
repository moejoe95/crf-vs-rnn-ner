import numpy as np
import pycrfsuite

import conll_parser
from FeatureGenerator import FeatureGenerator
import pos_tagger

# parse file
docs = conll_parser.parse("train.conll")
# do pos tagging, as part of feature extraction
data = pos_tagger.tag(docs)
feature = FeatureGenerator()
features = feature.extract_word_features(data)

# split test / train data
X_train, _, y_train, _ = feature.get_train_test_split(data, features)

# set up trainer
trainer = pycrfsuite.Trainer(verbose=True)

# Submit training data to the trainer
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

# Set the parameters of the model
trainer.set_params({
    'c1': 0.1,
    'c2': 0.01,  
    'feature.possible_transitions': True
})

# save model to file
trainer.train('crf.model')