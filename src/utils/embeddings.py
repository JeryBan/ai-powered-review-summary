
from pathlib import Path
import zipfile
import requests
import os

def download_weights():
    '''Downloads glove.6B.200d.txt , pretrained word weights
       and returns the path of the downloaded file.'''
    file_destination = Path('data/saved_models')
    
    print('Downloading weights...')
    with open('glove.6B.200d.zip', 'wb') as f:
        response = requests.get('https://storage.googleapis.com/kaggle-data-sets/13926/18767/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240229%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240229T114040Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4c1c59d7ff03b749bc5a3d74c5211a9dc203f9b3d4582ec867eb582e4ed7725d935ccb77dc1b7dc5b0216245492c765b0091c3519e266f750447a56cd34b5fa8c6de769c722c1ad37ddd891a9926ee87d7823d889da35922efbe07bfc4abf9991680951b4a246d3b3fc0b2ca7e2b6524d03f85a5718c2d04a0bf45150eb504707e64230862fb1439ea5ffab680cdff33c50b493e625702b4f8d594d2bc6a1c7116f4b63edf2fbb65eef09510dc9b9687997bfb331cdb414540b758db3d904d33dd129a09f41ece8e9223138fe6bff3ff62a77b8737dee24ff2175b37593121500445b822d863bbf2919cfcb1885947ab24e3a14d3afdcba79a211569379d9d44')
        f.write(response.content)

    print(f'Unziping to {file_destination} ...')
    with zipfile.ZipFile('glove.6B.200d.zip', 'r') as zip_ref:
        zip_ref.extractall(file_destination)

    os.remove('glove.6B.200d.zip')
    print('Done')

    return file_destination / 'glove.6B.200d.txt'

from gensim.corpora import Dictionary
import numpy as np
import torch

def get_embedding_matrix(weights_path):
    '''Populates a matrix with the pretrained weights for every word
       in the vocabulary present in the weights file.'''
    vocab = Dictionary.load('./data/saved_models/sentiment_vocab.pt')
    stoi = vocab.token2id
    
    embeddings_index = {}
    
    f = open(weights_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    embedding_matrix = np.zeros((len(stoi) + 1, 200))
    
    for word, i in stoi.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return torch.tensor(embedding_matrix, dtype=torch.float)
    
