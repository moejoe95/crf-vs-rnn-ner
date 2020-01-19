import nltk
import re as regex
import gensim 
from gensim.models import Word2Vec 
from WordShape import WordShape
from word2vec import word2vec
from BrownWrapper import BrownWrapper
from LDA import LDA

from nltk import cluster
from nltk.corpus import wordnet
from nltk.corpus import words
from nltk.corpus import names

class FeatureGenerator:

    stopwords = []
    w2v_dict = None
    tokens = []
    data = None
    ws = None
    brown_dict = None
    gazetteer = None
    frequency = None
    wordset = set(words.words())
    nameset = set(names.words())

    def __init__(self, data):
        self.data = data
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.ws = WordShape()

        # train and predict word2vec clustering
        print('training word2vec ...')
        for doc in data:
            for word in doc:
                self.tokens.append(word[0])
        w2v = word2vec(self.tokens, 5)
        w2v.train()
        self.w2v_dict = dict(zip(self.tokens, w2v.predict()))

        print('train brown clustering ...')
        brown_wrapper = BrownWrapper(data)
        self.brown_dict = brown_wrapper.get_brown_clustering()

        print('train LDA topic clustering ...')
        self.lda = LDA(self.tokens)

        print('\nextracting features ...\n')


    def get_features(self, doc, index):
        
        # word features
        word = doc[index][0]
        syns = wordnet.synsets(word)
        specificness = 0
        hypernym = 'nothing'
        if len(syns) > 0:
            specificness = len(syns[0].hypernym_paths()[0])
            if len(syns[0].hypernym_paths()) > 0 and len(syns[0].hypernym_paths()[0]) > 2:
                hypernym = syns[0].hypernym_paths()[0][2]
        feat = [
            'bias',
            'word.len=%s' % len(word),
            'word[-3:]=' + word[-3:],
            #'word[:3]=' + word[:3],
            'word.startsupper=%s' % word[0].isupper(),
            'word.containsdigit=%s' % any(char.isdigit() for char in word),
            'word.containsspecial=%s' % any((not char.isalnum()) for char in word),
            #'word.isfirst=%s' % (index == 0),
            #'word.islast=%s' % (index == len(doc)-1),
            #'word.isstopword=%s' % (word in self.stopwords),
            'word.shape=%s' % self.ws.get_wordshape(word),
            'word.w2vcluster=%s' % self.w2v_dict[word],
            'word.brownbitseq=%s' % self.brown_dict[word][0],
            'word.frequency=%s' % self.brown_dict[word][1],
            'word.browncluster=%s' % self.brown_dict[word][2],
            'word.ldatopic=%s' % self.lda.lda.get(word, -1), # TODO maybe implement per sentence instead of per word?
            'word.inwordlist=%s' % (word in self.wordset),
            'word.innamelist=%s' % (word in self.nameset),
            'word.specificness=%s' % specificness,
            #'word.hypernym=%s' % hypernym,
            'postag=' + doc[index][1]
        ]

        # context features
        if index > 0:
            feat.append('-1:postag=' + doc[index-1][1])
            feat.append('BOS=0')
        else:
            feat.append('BOS=1')

        if index < len(doc)-1:
            feat.append('+1:postag=' + doc[index+1][1])
            feat.append('EOS=0')
        else:
            feat.append('EOS=1')
        return feat 

    def word2features(self, doc, i):
        features = self.get_features(doc, i)
        return features


    # A function for extracting features in documents
    def extract_features(self, doc):
        return [self.word2features(doc, i) for i in range(len(doc))]


    # A function fo generating the list of labels for each document
    def get_labels(self, doc):
        return [label for (token, postag, label) in doc]


    def extract_word_features(self):
        return [self.extract_features(doc) for doc in self.data]
 