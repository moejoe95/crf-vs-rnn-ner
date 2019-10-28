import nltk

def tag(docs):
    data = []
    for i, doc in enumerate(docs):

        # Obtain the list of tokens in the document
        tokens = [t for t, label in doc]

        # Perform POS tagging
        #nltk.download('averaged_perceptron_tagger')
        tagged = nltk.pos_tag(tokens)

        # Take the word, POS tag, and its label
        data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])
    return data