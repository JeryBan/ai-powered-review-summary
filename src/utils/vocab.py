
from gensim.corpora import Dictionary
from typing import List, Dict

def create_vocabulary(document: List[str], min_freq: int) -> Dict[int, str]:
    '''Creates a vocabulary using the gensim library.'''
    special_tokens = {'<unk>': 1, '<pad>': 0}
    doc = [doc.split(' ') for doc in document]
    
    vocab = Dictionary(doc)
    vocab.patch_with_special_tokens(special_tokens)
    vocab.filter_extremes(no_below=min_freq)
    return vocab
