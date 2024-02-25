
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

import spacy
# python -m spacy download en_core_web_sm

import demoji
import re
from typing import List

class TextCleaner():
    '''Performs various transformation to a string to prepare it for nlp.
        
        Example usage:
            f = TextCleaner(remove_verbs=False)
            clean_text = f.clean('this is an example text')'''
    
    def __init__(self, remove_stopwords: bool = True, remove_verbs: bool = True, apply_lemma: bool = True):
        self.remove_verbs = remove_verbs
        self.remove_stopwords = remove_stopwords
        self.apply_lemma = apply_lemma
        self.tokens = []

        self.sp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    def tokenizer(self, text: str) -> List[str]:
        '''Transforms input text to lowercase and splits it to tokens'''
        doc = self.sp(text)
        tokens = []

        for token in doc:
            if self.remove_verbs and not token.pos_.startswith('N'):
                continue
        
            if self.remove_stopwords and token.text.lower() in stopwords.words('english'):
                continue

            # Check if the token is not a punctuation or whitespace and is not empty
            if not token.is_punct and not token.is_space and token.text.strip():          
                tokens.append(token.text.lower())
        return tokens

    def lemmatize(self, text: str) -> List[str]:
        '''Returns lemmatized tokens if apply_lemma = True'''
        doc = self.sp(text)
        tokens = []

        for token in doc:
            if self.remove_verbs and not token.pos_.startswith('N'):
                continue
        
            if self.remove_stopwords and token.text.lower() in stopwords.words('english'):
                continue
                
            # Check if the token is not a punctuation or whitespace and is not empty
            if not token.is_punct and not token.is_space and token.text.strip():
                lemma_token = token.lemma_.lower()
                tokens.append(lemma_token)
        return tokens

    def _demoji_replace(self, text: str) -> str:
        '''Replaces emojis with text'''
        emojis = demoji.findall(text)
        for emoji in emojis:
            text = text.replace(emoji, ' ' + emojis[emoji].split(':')[0])    
        return text

    def clean(self, text: str) -> str:
        '''Performs a full transformation of the input text'''
        # Remove urls
        clean_text = re.sub(r"http\S+", "", text)
        # Replace emojis
        clean_text = self._demoji_replace(clean_text)
        # Tokenize & lemmatization
        if self.apply_lemma:
            self.tokens = self.lemmatize(clean_text)
        else:
            self.tokens = self.tokenizer(clean_text)
            
        # Join tokens back into a single string
        cleaned_text = " ".join(self.tokens)
        # self.tokens = tokens
        return cleaned_text
