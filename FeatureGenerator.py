from sklearn.model_selection import train_test_split
import nltk
import re as regex
import gensim 
from gensim.models import Word2Vec 
from sklearn.cluster import DBSCAN
import WordShape

class FeatureGenerator:

    stopwords = []
    w2v_clusters = None
    tokens = []
    ws = None

    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.ws = WordShape.WordShape()


    def get_features(self, doc, index):
        word = doc[index][0]
        postag = doc[index][1]
        feat = [
            'bias',
            'word.len=%s' % len(word),
            'word[-3:]=' + word[-3:],
            'word[:3]=' + word[:3],
            'word.containsupper%s' % any(char.isupper() for char in word),
            'word.containsnonalpha=%s' % all((not char.isalpha()) for char in word),
            'word.isstopword=%s' % (word in self.stopwords),
            'word.shape=%s' % self.ws.get_wordshape(word),
            #'word.w2vcluster=%s' % self.w2v_clusters[index],
            'word.cfrequency=%s' % self.tokens.count(word),
            'postag=' + postag
        ]

        return feat


    def word2features(self, doc, i):
        # TODO
        # name lookup, dictionary
        # count vectorizer
        # brown clusters
        
        features = self.get_features(doc, i)
        return features


    # A function for extracting features in documents
    def extract_features(self, doc):
        return [self.word2features(doc, i) for i in range(len(doc))]


    # A function fo generating the list of labels for each document
    def get_labels(self, doc):
        return [label for (token, postag, label) in doc]

    def train_w2v(self, data):
        for doc in data:
            for word in doc:
                self.tokens.append(word[0])
        w2v = gensim.models.Word2Vec(
            self.tokens,
            size=150,
            window=10,
            min_count=2,
            workers=10,
            iter=10) 
        dbscan = DBSCAN(metric='cosine', eps=0.07)
        self.w2v_clusters = dbscan.fit_predict(w2v.wv.vectors) 


    def extract_word_features(self, data):
        self.train_w2v(data)
        return [self.extract_features(doc) for doc in data]
 