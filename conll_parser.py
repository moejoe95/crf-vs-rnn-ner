# parser for conll files

def parse(filename):
    with open(filename, "r") as f:
        docs = []
        doc = []
        for line in f:
            word_label = line.split("\t")
            if len(word_label) < 2 or len(word_label[0].strip()) == 0:
                docs.append(doc)
                doc = []
                continue
            doc.append((word_label[0].strip(), word_label[1].strip()))
    return docs