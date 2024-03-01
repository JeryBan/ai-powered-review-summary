
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary

import os
from typing import List, Dict

def extract_topics(document: List[str],
                    num_topics: int = 3,
                    passes: int = 1,
                    iterations: int = 100) -> Dict[int, str]:
    '''Initializes and trains the lda model.
       Returns the top n topics of the document.'''

    vocab = Dictionary.load('./data/saved_models/topics_vocab.pt')
    corpus = [vocab.doc2bow(text.split(' ')) for text in document]

    num_cores = os.cpu_count()

    # initialize and train lda model
    lda_model = LdaMulticore(corpus=corpus, id2word=vocab,
                             num_topics=1,
                             passes=passes,
                             iterations=iterations,
                             workers=num_cores)
                             
    lda_model.save('data/saved_models/lda_model')

    # get the top n topics
    labels = {id: topic[0] for id, topic in enumerate(lda_model.show_topic(0, topn=num_topics))}

    return labels
                        
