import numpy as np
from gensim.models import KeyedVectors
from sklearn.externals import joblib

from settings import PATH_TO_EMBEDDINGS, PATH_TO_SCALER


class Embeddings():

    def __init__(self, textpreprocess):
        self.embeddings = KeyedVectors.load_word2vec_format(str(PATH_TO_EMBEDDINGS))
        self.embeddings_dim = self.embeddings.vector_size
        self.scaler = joblib.load(str(PATH_TO_SCALER))
        self.textpreprocess = textpreprocess

    def get_vector_embeddings(self, tokens):
        X = np.zeros((self.embeddings_dim), dtype=np.float32)
        empty_tokens = []
        tokens_embeddings = []
        for t in tokens:
            try:
                tokens_embeddings.append(self.embeddings.get_vector(t))
            except KeyError:
                tokens_embedding = self.get_embeddings_of_error_token(t, tokens_embeddings, empty_tokens)
        if len(tokens_embeddings) > 0:
            mean_embeddings = np.mean(tokens_embeddings, axis=0)
            X = mean_embeddings
        if len(empty_tokens) > 0:
            print(f'Empty tokens: {empty_tokens}')
        return X

    def get_embeddings_of_error_token(self, token, tokens_embeddings, empty_tokens):
        tokens = self.textpreprocess.tokenize_with_lower(token)
        for t in tokens:
            try:
                tokens_embeddings.append(self.embeddings.get_vector(t))
            except KeyError:
                empty_tokens.append(t)
        return tokens_embeddings
