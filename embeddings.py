import os
import numpy as np
from keras.layers import Embedding

def getPreTrainedEmbeddingLayer(word2idx, wordlen, vecfile, veclen, max_len):
    embeddings_index = {}
    f = open(vecfile)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((wordlen + 1, veclen))
    for word, i in word2idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return Embedding(wordlen + 1, veclen, weights=[embedding_matrix], input_length=max_len)
