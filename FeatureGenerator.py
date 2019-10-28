from sklearn.model_selection import train_test_split

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
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
            'word.isstopword=%s' % (word in self.stopwords),
            'word.containshyphen=%s' % ('-' in word),
            'word.containsunderscore=%s' % ('_' in word), 
            'postag=' + postag
        ]

    def word2features(self, doc, i):
        # TODO
        # word contains digit
        # upper case in middle of word
        # special characters in word
        # name lookup, dictionary
        # global information like frequency
        # count vectorizer

        # Common features for all words
        features = self.get_features(doc, i)

        # Features for words that are not
        # at the beginning of a document
        if i > 0:
            features.extend(self.get_features(doc, i-1))
        else:
            # Indicate that it is the 'beginning of a document'
            features.append('BOS')

        # Features for words that are not 
        # at the end of a document
        if i < len(doc)-1:
            features.extend(self.get_features(doc, i+1))
        else:
            # Indicate that it is the 'end of a document'
            features.append('EOS')

        return features


    # A function for extracting features in documents
    def extract_features(self, doc):
        return [self.word2features(doc, i) for i in range(len(doc))]


    # A function fo generating the list of labels for each document
    def get_labels(self, doc):
        return [label for (token, postag, label) in doc]

    def extract_word_features(self, data):
        return [self.extract_features(doc) for doc in data]

    def get_train_test_split(self, data, features):
        labels = [self.get_labels(doc) for doc in data]
        return train_test_split(features, labels, test_size=0.25)
