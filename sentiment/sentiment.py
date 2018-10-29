import numpy as np

from sentiment.initialize import textpreprocess, embeddings, mlmodel


def get_sentiment(text):
    text_tok = textpreprocess.tokenize_only(text)
    text_vec = embeddings.get_vector_embeddings(text_tok)
    text_vec_scale = embeddings.scaler.transform(text_vec.reshape(1, embeddings.embeddings_dim))
    pred = mlmodel.predict(text_vec_scale)[0]
    max_pred = max(pred)
    max_pred_position = np.where(pred == max_pred)[0][0]
    return mlmodel.mlb.classes_[max_pred_position]
