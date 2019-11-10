from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import PorterStemmer
import re as regex
import gensim 
from gensim.models import Word2Vec 
from sklearn.cluster import DBSCAN

class FeatureGenerator:

    stopwords = []
    stemmer = PorterStemmer()
    w2v_clusters = None
    data = None

    def __init__(self):
        #nltk.download('stopwords')
        self.stopwords = nltk.corpus.stopwords.words('english')

    def get_features(self, doc, index):
        word = doc[index][0]
        postag = doc[index][1]
        feat = [
            'bias',
            'word.lower=' + word.lower(),
            'word.stem=' + self.stemmer.stem(word),
            # prefixes up to length of 4
            'word[-4:]=' + word[-4:],
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
            'word.isstopword=%s' % (word in self.stopwords),
            'word.shape=%s' % self.wordshape(word),
            'word.w2vcluster=%s' % self.w2v_clusters[index],
            'postag=' + postag
        ]
        feat += self.len_index(word, index)
        feat += self.contains(word)
        feat += self.collection_features(word)
        return feat

    def word2features(self, doc, i):
        # TODO
        # name lookup, dictionary
        # count vectorizer
        # brown clusters
        
        features = self.get_features(doc, i)
        return features

    def collection_features(self, word):
        return [
            'word.cfrequency=%s' % self.data.count(word)
        ]

    def len_index(self, word, index):
        return [
        'word.len=%s' % len(word),
        'word.index=%s' % index
        ]

    def contains(self, word):
        return [
            'word.containshyphen=%s' % ('-' in word),
            'word.containsunderscore=%s' % ('_' in word), 
            'word.containsdigit=%s' % any(char.isdigit() for char in word),
            'word.containsupper%s' % any(char.isupper() for char in word),
            'word.containsspecialchar=%s' % regex.match('^[a-zA-Z0-9]+$', word)
            ]

    # from https://stackoverflow.com/questions/49945812/is-there-any-word-shape-feature-library-for-ner-in-python
    def wordshape(self, text):
        # todo: try to remove consecutive chars
        t1 = regex.sub('[A-Z]', 'X',text)
        t2 = regex.sub('[a-z]', 'x', t1)
        t3 = regex.sub('[^a-zA-Z\d]', '-', t2) # special char
        return regex.sub('[0-9]', 'd', t3)

    # A function for extracting features in documents
    def extract_features(self, doc):
        return [self.word2features(doc, i) for i in range(len(doc))]

    # A function fo generating the list of labels for each document
    def get_labels(self, doc):
        return [label for (token, postag, label) in doc]

    def extract_word_features(self, data):
        self.data = data
        tokens = []
        for doc in data:
            for word in doc:
                tokens.append(word[0])
        w2v = gensim.models.Word2Vec(
            tokens,
            size=150,
            window=10,
            min_count=2,
            workers=10,
            iter=10) 
        w2v_vectors = w2v.wv.vectors 
        dbscan = DBSCAN(metric='cosine', eps=0.07, min_samples=3)
        self.w2v_clusters = dbscan.fit_predict(w2v_vectors) 
        return [self.extract_features(doc) for doc in data]
 