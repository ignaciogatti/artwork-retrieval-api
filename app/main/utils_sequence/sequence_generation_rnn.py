import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances,euclidean_distances
from .Prediction_model_feature import Prediction_model_feature
from .abstract_sequence_rnn import Abstract_sequence_rnn


BASE_MODEL_DIR = os.path.join(os.getcwd(), 'static/model')
MODEL_DIR = os.path.join(BASE_MODEL_DIR, 'sequence_RNN')



WINDOW_INDEX = 3
#Sequence RNN model
museum_sequence_path = {
    'x_test' : os.path.join(MODEL_DIR, 'X_test.csv'),
    'x_test_matrix' : os.path.join(MODEL_DIR, 'X_test_matrix.npy'),
    'weights_folder' : os.path.join(MODEL_DIR, 'config_'+str(WINDOW_INDEX)+'/trained_model_weights'),
    'all_metadata' : os.path.join(BASE_MODEL_DIR, 'train_mayors_style_encoded_with_url.csv'),
    'all_data_matrix' : os.path.join(BASE_MODEL_DIR, 'train_mayors_style_encode.npy' )
}


class Sequence_generator_rnn(Abstract_sequence_rnn):
    
    def __init__(self, batch_size=128, shuffle_buffer_size=300, conv_filter=16, lstm_filter=32, dense_filter=16, prediction_length=10):
        super().__init__(WINDOW_INDEX, museum_sequence_path, batch_size, shuffle_buffer_size, conv_filter, lstm_filter, dense_filter, prediction_length)
        self._model = self._load_model()
    
    
    def _create_rnn_model(self):
        return Prediction_model_feature(
                X=self._X[:, 0],
                train_batch_size=self._batch_size, 
                val_batch_size=self._batch_size, 
                window_size=self._window_size, 
                shuffle_buffer=self._shuffle_buffer_size
                )
        