
import torch
from gensim.models import LdaMulticore
from src.utils.vocab import CustomVocab

import os
from typing import List, Dict

def extraxt_topics(document: List[str],
                    num_topics: int = 3,
                    passes: int = 1,
                    iterations: int = 100) -> Dict[int, str]:
    '''Initializes and trains the lda model.
       Returns the top n topics of the document.'''

    vocab = CustomVocab(document)
    id2word = vocab.id2word
    bow = vocab.bow
    

    num_cores = os.cpu_count()

    # initialize and train lda model
    lda_model = LdaMulticore(corpus=bow, id2word=id2word,
                             num_topics=1,
                             passes=passes,
                             iterations=iterations,
                             workers=num_cores)
                             
    lda_model.save('data/saved_models/lda_model')

    # get the top n topics
    labels = {id: topic[0] for id, topic in enumerate(lda_model.show_topic(0, topn=num_topics))}

    return labels
                        
