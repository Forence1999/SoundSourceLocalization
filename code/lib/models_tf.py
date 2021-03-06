import tensorflow as tf
from .model_tf.FCN import *
from .model_tf.EEGNet import *
from .model_tf.ShallowCNN import *
from .model_tf.DeepCNN import *
from .model_tf.LeNet5 import *
from .model_tf.TSLeNet5 import *
from .model_tf.RD3Net import *
from .model_tf.ResCNN_4_STFT_DOA import *

# # define the classes of DTW and soft-DTW-Probabilistic
# class DTW(BaseClassicalModel):
#
#     def __init__(self, *args, neighbours=1, name=None, **kwargs):
#         super(DTW, self).__init__(name=name, **kwargs)
#
#         self.neighbours = neighbours
#         self.model = DTW.KnnDTW(neighbours)
#
#     def fit(self, X, Y, training=True, **kwargs):
#         self.model.fit(X, Y)
#
#     def predict(self, X, training=False, **kwargs):
#         mode_label, mode_proba = self.model.predict(X)
#         return mode_label
#
#
# class DTWProbabilistic(BaseClassicalModel):
#
#     def __init__(self, *args, name=None, **kwargs):
#         if name is None:
#             name = self.__class__.__name__
#
#         super(DTWProbabilistic, self).__init__(name=name, **kwargs)
#
#         self.neighbours = 1
#         self.model = DTW.KnnDTW(self.neighbours)
#
#     def fit(self, X, Y, training=True, **kwargs):
#         self.model.fit(X, Y)
#
#     def predict(self, X, training=False, **kwargs):
#         probas, labels = self.model.predict_proba(X)
#         return probas
#
