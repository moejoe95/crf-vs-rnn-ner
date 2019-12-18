# parser for conll files
import re
import random

def parse(filename):
    words = []
    labels = []
    docs = []
    with open(filename, "r") as f:
        w = []
        l = []
        doc = []
        for line in f:
            word_label = re.split(' |\t', line)
            if len(word_label) < 2 or len(word_label[0].strip()) == 0:
                words.append(w)
                labels.append(l)
                docs.append(doc)
                w = []
                l = []
                doc = []
                continue
            doc.append((word_label[0].strip(), word_label[len(word_label)-1].strip()))
            w.append(word_label[0].strip())
            l.append(word_label[len(word_label)-1].strip())
    return docs, words, labels


def filter_parsed(docs, rand):
    rand_docs = []
    count = len(docs)
    rand_lines = []
    for i in range(0, int(rand)):
        rand_lines.append(random.randint(0, count))
    for i in rand_lines:
        rand_docs.append(docs[i])
    return rand_docs
