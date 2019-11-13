from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class LDA:

    lda = dict()

    def __init__(self, tokens):
        count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
        doc_term_matrix = count_vect.fit_transform(tokens)
        LDA = LatentDirichletAllocation(n_components=3, random_state=42)
        LDA.fit(doc_term_matrix)

        for i,voc_entry in enumerate(count_vect.vocabulary_):
            max_topic = 0
            max_index = 0
            for j,topic in enumerate(LDA.components_):
                if (topic[i] > max_topic):
                    max_topic = topic[i]
                    max_index = j
            self.lda.update({voc_entry: max_index})

