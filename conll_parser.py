# parser for conll files
import re

def parse(filename):
    words = []
    docs = []
    with open(filename, "r") as f:
        w = []
        doc = []
        for line in f:
            word_label = re.split(' |\t', line)
            if len(word_label) < 2 or len(word_label[0].strip()) == 0:
                words.append(w)
                docs.append(doc)
                w = []
                doc = []
                continue
            doc.append((word_label[0].strip(), word_label[len(word_label)-1].strip()))
            w.append(word_label[0].strip())
    return docs, words