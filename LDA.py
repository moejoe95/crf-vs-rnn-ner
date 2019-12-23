from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pyLDAvis import sklearn as sklearn_lda
import pickle 
import pyLDAvis
import os

class LDA:

    lda = dict()
    clusters = 3
    i = 0

    def __init__(self, tokens):
        count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
        doc_term_matrix = count_vect.fit_transform(tokens)
        LDA = LatentDirichletAllocation(
            n_components=self.clusters, 
            random_state=42, 
            learning_method='online', 
            max_iter=5, 
            learning_offset=50
        )
        LDA.fit(doc_term_matrix)

        '''
        LDAvis_data_filepath = os.path.join('./ldavis_prepared_'+str(self.clusters))
        LDAvis_prepared = sklearn_lda.prepare(LDA, doc_term_matrix, count_vect)

        with open(LDAvis_data_filepath, 'r+b') as f:
            pickle.dump(LDAvis_prepared, f)
                
        # load the pre-prepared pyLDAvis data from disk
        with open(LDAvis_data_filepath, 'rb') as f:
            LDAvis_prepared = pickle.load(f)
            pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_'+ str(self.clusters) +'.html')
        '''
        for i,voc_entry in enumerate(count_vect.vocabulary_):
            max_topic = 0
            max_index = 0
            for j,topic in enumerate(LDA.components_):
                if (topic[i] > max_topic):
                    max_topic = topic[i]
                    max_index = j
            self.lda.update({voc_entry: max_index})

