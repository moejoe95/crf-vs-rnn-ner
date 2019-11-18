#!/usr/bin/env python
"""crf_train, a tool to train the CRF NER system with a given file.

Usage:
  crf_train.py (-t | --train) <file> (-m | --model) <model>
  crf_train.py (-h | --help)

Options:
  -h --help     Show this screen.
"""

from docopt import docopt
import numpy as np
import pycrfsuite

import conll_parser
from FeatureGenerator import FeatureGenerator
import pos_tagger
import os

arguments = docopt(__doc__, version='crf_train')
train_file = arguments.get('<file>', './data/conll/eng.train')
model = arguments.get('<model>', 'crf.model')

# parse file
docs = conll_parser.parse(train_file)
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
    'c1': 0.5,
    'c2': 0.01,  
    'feature.possible_transitions': True
})

# save model to file
trainer.train(model)
