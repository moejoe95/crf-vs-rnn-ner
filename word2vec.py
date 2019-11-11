import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import MiniBatchKMeans
import gensim
import os


class word2vec:

    K = None
    X = None
    model = None 
    tokens = None
    classifier = None

    def __init__(self, tokens, K):
        self.K = K
        self.tokens = tokens
        if os.path.isfile('w2v.model'):
            self.model = Word2Vec.load('w2v.model')
        else:
            self.model = Word2Vec(self.tokens) 
            self.model.save('w2v.model')
        self.make_dataset()


    def make_dataset(self):
        V = self.model.wv.index2word
        X = np.zeros((len(V), self.model.wv.vector_size))

        for index, word in enumerate(V):
            X[index, :] += self.model[word]

        self.X = X


    def train(self):
        self.classifier = MiniBatchKMeans(n_clusters=self.K, random_state=0)
        self.classifier.fit(self.X)


    def predict(self):
        X = [self.model[word] for word in self.tokens if word in self.model]
        classes = self.classifier.predict(X)

        result = []
        i = 0
        for word in self.tokens:
            if word in self.model:
                result.append(str(classes[i]))
                i += 1
            else:
                result.append(str(-1))
        return result
