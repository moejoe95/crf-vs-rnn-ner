from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def get_padded_seq(docs):
    words = []
    labels = []
    max_len = 0
    for sentence in docs:
        for word in sentence:
            words.append(word[0])
            labels.append(word[1])
        if len(sentence) > max_len:
            max_len = len(sentence)

    word2idx = {w: i for i, w in enumerate(words)}
    label2idx = {t: i for i, t in enumerate(labels)}

    X = [[word2idx[word[0]] for word in sentence] for sentence in docs]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=len(words) - 1)

    y = [[label2idx[word[1]] for word in sentence] for sentence in docs]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=label2idx["O"])

    y = [to_categorical(i, num_classes=len(labels)) for i in y]

    return X, y, len(labels), max_len
