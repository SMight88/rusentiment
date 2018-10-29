import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from sklearn.externals import joblib

from settings import PATH_TO_MODEL, PATH_TO_MLB


class MLModel():

    def __init__(self):
        self.mlb = joblib.load(str(PATH_TO_MLB))
        self.model, self.graph = self.load_model()

    def load_model(self):
        model = load_model(str(PATH_TO_MODEL), custom_objects={'precision': self.precision,
                                                               'recall': self.recall,
                                                               'f1score': self.f1score})
        graph = tf.get_default_graph()
        return model, graph

    def predict(self, vector):
        with self.graph.as_default():
            return self.model.predict(vector)

    def precision(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def fbeta(self, y_true, y_pred, beta=1):
        if beta < 0:
            raise ValueError('The lowest choosable beta is zero (only precision).')

        # If there are no true positives, fix the F score at 0 like sklearn.
        if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
            return 0

        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        bb = beta ** 2
        fbeta = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
        return fbeta

    def f1score(self, y_true, y_pred):
        return self.fbeta(y_true, y_pred, beta=1)
