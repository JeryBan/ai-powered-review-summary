
import torchtext
from torchtext.vocab import vocab
import gensim.corpora as corpora

from collections import Counter, OrderedDict
from typing import List, Dict, Union

class CustomVocab(torchtext.vocab.Vocab):
    '''Creates a custom vocabulary from a list of strings and provides various information 
        about it in the form of attributes.'''
    
    def __init__(self, document: Union[List[str], str]):
        super(CustomVocab, self).__init__(None)
        
        self.rawText = document
        self.tokens = []
        self.word_freqs = []
        self.vocab = self._create_vocab(document)
        self.size = len(self.word_freqs)
        self.id2word = []
        self.bow = self._bag_of_words(document)

    def __len__(self):
        return len(self.word_freqs)

    def _create_vocab(self, document):
        tokens = self._get_tokens(document)
        orderedDict = self._get_word_freq(tokens)
        
        vocab = torchtext.vocab.vocab(ordered_dict=orderedDict, min_freq=1)
        vocab.set_default_index(-1)
        return vocab

    def _get_tokens(self, document):
        if isinstance(document, str):
            document = [document]

        tokens = []
        for word in document:
            token = word.split(' ')
            tokens.extend(token)

        self.tokens = tokens
        return tokens

    def _get_word_freq(self, tokens):       

        counter = Counter(tokens)
        sort_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.word_freqs = sort_counter
        return OrderedDict(counter)

    def _bag_of_words(self, document):
        words = []
        for doc in document:
            words.append([token for token in doc.split(' ')])
    
        self.id2word = corpora.Dictionary(words)
        
        return [self.id2word.doc2bow(word) for word in words]
