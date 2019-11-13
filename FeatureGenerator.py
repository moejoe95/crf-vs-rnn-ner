import nltk
import re as regex
import gensim 
from gensim.models import Word2Vec 
from WordShape import WordShape
from word2vec import word2vec
from BrownWrapper import BrownWrapper
from Gazetteer import Gazetteer
from LDA import LDA
from nltk import cluster

class FeatureGenerator:

    stopwords = []
    w2v_dict = None
    tokens = []
    data = None
    ws = None
    brown_dict = None
    gazetteer = None
    frequency = None

    def __init__(self, data):
        self.data = data
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.ws = WordShape()

        # train and predict word2vec clustering
        for doc in data:
            for word in doc:
                self.tokens.append(word[0])
        w2v = word2vec(self.tokens, 5)
        w2v.train()
        self.w2v_dict = dict(zip(self.tokens, w2v.predict()))

        brown_wrapper = BrownWrapper(data)
        self.brown_dict = brown_wrapper.get_brown_clustering()
        self.gazetteer = Gazetteer(data)

        self.lda = LDA(self.tokens)


    def get_features(self, doc, index):
        word = doc[index][0]
        postag = doc[index][1]
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
            #'word.gazetteer=%s' % self.gazetteer.gazetteer.get(word, 0),
            'word.ldatopic=%s' % self.lda.lda.get(word, -1),
            'postag=' + postag
        ]

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
 