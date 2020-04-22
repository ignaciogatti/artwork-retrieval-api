import tensorflow as tf
import numpy as np
import pandas as pd
import os
from abc import abstractmethod, ABC
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances,euclidean_distances
from ..utils_sequence.Prediction_model_feature import Prediction_model_feature




class Abstract_sequence_rnn(ABC):
    
    def __init__(self, window_size, museum_sequence_path, batch_size, shuffle_buffer_size, conv_filter=16, lstm_filter=32, dense_filter=16, prediction_length=1):
        self._name= "Sequence_generator_rnn"
        self._museum_sequence_path = museum_sequence_path
        self._window_size = window_size
        self._df_all_metadata = pd.read_csv(self._museum_sequence_path['all_metadata'])
        self._all_data_matrix = np.load(self._museum_sequence_path['all_data_matrix'])
        self._X = np.load(self._museum_sequence_path['x_test_matrix'])
        self._batch_size = batch_size
        self._shuffle_buffer_size = shuffle_buffer_size
        self._conv_filter = conv_filter
        self._lstm_filter = lstm_filter
        self._dense_filter = dense_filter
        self._prediction_length = prediction_length
        

    @abstractmethod    
    def _create_rnn_model(self):
        pass
    
    
    def _load_model(self):
        self._n_features = self._X.shape[1]
        #Create model
        self._model = self._create_rnn_model()
        self._model.define_model(
            conv_filter=self._conv_filter, 
            lstm_filter=self._lstm_filter, 
            dense_filter=self._dense_filter, 
            prediction_length=self._prediction_length
            )
        return self._model
    
    
    
    def _drop_selected_artwork(self, indexes, df_all_metadata, all_data_matrix):
    
        #Remove from metadata
        df_removed = df_all_metadata.copy()
        df_removed = df_removed.drop(indexes)
        df_removed = df_removed.reset_index(drop=True)

        #Remove ftom code matrix
        code_matrix = all_data_matrix.copy()
        code_matrix = np.delete(code_matrix, indexes, 0)

        return df_removed, code_matrix
    

    def _predict_features(self):
        
        predicted_features = []
        for feature in range(self._n_features):
            #Load weights for feature i
            self._model.set_index(feature)
            self._model.load_weights(self._museum_sequence_path)

            #Reshape to be a valid input for the model
            x_feature = self._X_tour[:,feature]
            x_feature = tf.expand_dims(x_feature, axis=-1)
            x_feature = tf.expand_dims(x_feature, axis=0)
    
            #Predict feature i
            rnn_forecast = self._model.get_model().predict(x_feature)
            rnn_forecast = rnn_forecast.reshape((-1))
            
            predicted_features.append(rnn_forecast)
        
        self._forecast_matrix = np.stack(predicted_features)
        self._forecast_matrix = self._forecast_matrix.T
        return self._forecast_matrix
        
    
    def predict_tour(self):
        
        
        #Dataframe with the tour
        self._df_predicted_tour = pd.DataFrame(
            { 'id' : [],
              'title' : [],
              'artist' : [],
              'sim_value' : [],
              'imageUrl':[]})
       
        ##List with the artworks's code that belongs to the tour
        self._predicted_code_list =[]

        
                
        #Made a copy of the data to keep the data safe
        df_all_metadata = self._df_all_metadata.copy()
        all_data_matrix = self._all_data_matrix.copy()
        
        
        #Predict features
        self._forecast_matrix = self._predict_features()

        for i in range(self._forecast_matrix.shape[0]):
            #Find code
            code = self._forecast_matrix[i].reshape((1,-1))

            #Compute cosine similarity
            sim_matrix = cosine_similarity(code, all_data_matrix)

            #sort indexes
            sort_index = np.argsort(sim_matrix.reshape((-1,)))

            #Find most similar
            sim_artwork_index = sort_index[-1]

            #Save in dataframe 
            self._df_predicted_tour = self._df_predicted_tour.append(
                {'id' : int(df_all_metadata.iloc[sim_artwork_index].name),
                 'title' : df_all_metadata.iloc[sim_artwork_index]['title'],
                 'artist': df_all_metadata.iloc[sim_artwork_index]['artist'],
                 'imageUrl':df_all_metadata.iloc[sim_artwork_index]['imageUrl'],
                 'sim_value':sim_matrix[:,sim_artwork_index][0]
                }, 
               ignore_index=True)

            #Save predicted artwork's code
            self._predicted_code_list.append(all_data_matrix[sim_artwork_index])

            #Remove selected artworks
            df_all_metadata, all_data_matrix = self._drop_selected_artwork([sim_artwork_index], df_all_metadata, all_data_matrix)

        self._df_predicted_tour = self._df_predicted_tour.set_index('id')
        return self._df_predicted_tour
    

    def get_predicted_tour_matrix(self):
        #No tour predicted because the window size was too big
        if len(self._predicted_code_list) == 0:
            return np.array([])
        
        forecast_matrix = np.stack(self._predicted_code_list)
        return forecast_matrix
   

    def get_name(self):
        return self._name
    
    
    def get_model(self):
        return self._model
    
    
    def set_tour(self, X_tour):
        self._X_tour = X_tour

        