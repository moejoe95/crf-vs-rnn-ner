from sklearn.model_selection import train_test_split
import re as regex

class FeatureGenerator:

    stopwords = []

    def __init__(self):
        with open("stopwords.txt", "r") as f:
            for line in f:
                self.stopwords.append(line)

    def get_features(self, doc, index):
        word = doc[index][0]
        postag = doc[index][1]
        return [
            'bias',
            'word.lower=' + word.lower(),
            # prefixes up to length of 4
            'word[-4:]=' + word[-4:],
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
            'word.isstopword=%s' % (word in self.stopwords),
            'word.shape=%s' % self.wordshape(word),
            'postag=' + postag
        ]

    def word2features(self, doc, i):
        # TODO
        # name lookup, dictionary
        # global information like frequency
        # count vectorizer
        # brown clusters
        
        features = self.get_features(doc, i)
        return features

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
        return regex.sub('[0-9]', 'd', t2)

    # A function for extracting features in documents
    def extract_features(self, doc):
        return [self.word2features(doc, i) for i in range(len(doc))]

    # A function fo generating the list of labels for each document
    def get_labels(self, doc):
        return [label for (token, postag, label) in doc]

    def extract_word_features(self, data):
        return [self.extract_features(doc) for doc in data]
