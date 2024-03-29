
from gensim.corpora import Dictionary
from gensim.models import FastText

from typing import List, Dict
import numpy as np
import os

def train_model(document: List[str]):
    '''Initialize and train a FastText model.'''
    vocab = Dictionary.load('./data/saved_models/topics_vocab.pt')
    freqs = {vocab[k]: v for k, v in vocab.cfs.items()}
    # initialize
    fasttext_model = FastText(vector_size=100, window=4, min_count=1, workers=os.cpu_count(), sg=1)
    fasttext_model.build_vocab_from_freq(word_freq=freqs)
    # train & save model
    fasttext_model.train(corpus_iterable=reviews, total_examples=len(reviews), epochs=20)
    # fasttext_model.save('data/saved_models/fastText_model')

    return fasttext_model


def match_topics(document: List[str], labels: Dict[int, str]) -> Dict[str, str]:
    '''Matches each review to the most relevant topic.'''
    topic_dict = {}
    for row, review in enumerate(reviews):
        prob = []
        for _, topic in labels.items():
            prob.append(fasttext_model.wv.relative_cosine_similarity(topic, review))
            topic_dict[review] = labels[np.argmax(prob)]
            
    return topic_dict
