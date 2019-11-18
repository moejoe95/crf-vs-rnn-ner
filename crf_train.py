import numpy as np
import pycrfsuite

import conll_parser
from FeatureGenerator import FeatureGenerator
import pos_tagger
import os

# parse file
docs = conll_parser.parse("train.conll")
# do pos tagging, as part of feature extraction
data = pos_tagger.tag(docs)
feature = FeatureGenerator(data)
features = feature.extract_word_features()
labels = [feature.get_labels(doc) for doc in data]

# set up trainer
trainer = pycrfsuite.Trainer(verbose=True)

# Submit training data to the trainer
for xseq, yseq in zip(features, labels):
    trainer.append(xseq, yseq)

# Set the parameters of the model
trainer.set_params({
    'c1': 0.1,
    'c2': 0.01,  
    'feature.possible_transitions': True
})

# save model to file
os.remove('crf.model')
trainer.train('crf.model')
