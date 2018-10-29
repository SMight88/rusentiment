from sentiment.mlmodel import MLModel
from sentiment.textpreprocess import TextPreprocess
from sentiment.embeddings import Embeddings

mlmodel = MLModel()
textpreprocess = TextPreprocess()
embeddings = Embeddings(textpreprocess)