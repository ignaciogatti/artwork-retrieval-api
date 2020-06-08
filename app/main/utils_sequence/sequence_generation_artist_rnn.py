import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances,euclidean_distances
from .Prediction_model_feature import Prediction_model_feature
from .abstract_sequence_rnn import Abstract_sequence_rnn


BASE_MODEL_DIR = os.path.join(os.getcwd(), 'static/model')
CONFIG_DIR = os.path.join(BASE_MODEL_DIR, 'sequence_RNN')
SEQUENCE_MODEL_DIR = os.path.join(CONFIG_DIR, 'artist_code_embedding')



WINDOW_INDEX = 3
#Sequence RNN model
museum_sequence_path = {
    'x_test' : os.path.join(CONFIG_DIR, 'X_test.csv'),
    'x_test_matrix' : os.path.join(CONFIG_DIR, 'X_artist_embedding_test_matrix.npy'),

    'weights_folder' : os.path.join(SEQUENCE_MODEL_DIR, 'config_'+str(WINDOW_INDEX)+'/trained_model_weights'),

    'all_artist_metadata' : os.path.join(BASE_MODEL_DIR, 'all_artists.csv'),
    'all_artist_code_matrix' : os.path.join(BASE_MODEL_DIR, 'all_artists_code_matrix.npy')
}


class Sequence_generator_artist_rnn(Abstract_sequence_rnn):
    
    def __init__(self, df_all_metadata, all_data_matrix, batch_size=128, shuffle_buffer_size=300, conv_filter=20, lstm_filter=40, dense_filter=20, prediction_length=15):
        super().__init__(WINDOW_INDEX, museum_sequence_path, batch_size, shuffle_buffer_size, df_all_metadata, all_data_matrix, conv_filter, lstm_filter, dense_filter, prediction_length)
        print(self._X.shape)
        self._model = self._load_model()

        #Load artist data 
        self._df_all_artists = pd.read_csv(museum_sequence_path['all_artist_metadata'])
        self._all_artist_code_matrix = np.load(museum_sequence_path['all_artist_code_matrix'])
    
    
    def _create_rnn_model(self):
        return Prediction_model_feature(
                X=self._X,
                train_batch_size=self._batch_size, 
                val_batch_size=self._batch_size, 
                window_size=self._window_size, 
                shuffle_buffer=self._shuffle_buffer_size
                )


            
    def _define_x_features(self, feature):
        x_feature = self._X_tour[:,feature]
        x_feature = tf.expand_dims(x_feature, axis=-1)
        x_feature = tf.expand_dims(x_feature, axis=0)
        return x_feature   


    
    def _get_most_similar_artist(self, p):
    
        #Find nearest value. Try to take a couple
        nearest_index_sort = np.abs(self._all_artist_code_matrix - p).argsort()

        #Find most similar
        return list(self._df_all_artists.iloc[nearest_index_sort[:50]]['author'].values)    
    

    '''
    def _get_most_similar_artist(self, p):

        #Find nearest value. Try to take a couple
        nearest_index_sort = np.abs(self._all_data_matrix[:,-1] - p).argsort()

        #Find most similar
        return nearest_index_sort[:100]    
    '''

    def predict_tour(self):
        
        
        #Dataframe with the tour
        self._df_predicted_tour = pd.DataFrame(
            { 'id' : [],
              'title' : [],
              'artist' : [],
              'sim_value' : [],
              'imageUrl':[]})
       
        #List with the artworks's code that belongs to the tour
        self._predicted_code_list =[]

        #List with the artwokk's indexes already appended
        indexes_appended_list = []
                
        #Made a copy of the data to keep the data safe
        df_all_metadata = self._df_all_metadata.copy()
        all_data_matrix = self._all_data_matrix.copy()
        
        
        #Predict features
        self._forecast_matrix = self._predict_features()
        
        
        for i in range(self._forecast_matrix.shape[0]):
            #Find code
            code = self._forecast_matrix[i]

            #Define a valid subset
            
            artists = self._get_most_similar_artist(code[-1])
            artists_indexes = self._df_all_metadata[self._df_all_metadata['artist'].isin(artists)].index

            #artists_indexes = self._get_most_similar_artist(code[-1])

            #Drop artwork's indexes that already are part of the tour
            for index in indexes_appended_list:
                artists_indexes = artists_indexes[artists_indexes != index]


            artist_work_matrix =self._all_data_matrix[artists_indexes]    

            #Compute cosine similarity
            #The last feature is artist feature
            sim_matrix = cosine_similarity(code[:-1].reshape((1,-1)), artist_work_matrix[:,:-1])
            

            #sort indexes
            sort_index = np.argsort(sim_matrix.reshape((-1,)))

            #Find most similar
            sim_artwork_index = sort_index[-1]

            #Save in dataframe 
            self._df_predicted_tour = self._df_predicted_tour.append(
                {'id' : int(self._df_all_metadata.iloc[sim_artwork_index].name),
                 'title' : self._df_all_metadata.iloc[sim_artwork_index]['title'],
                 'artist': self._df_all_metadata.iloc[sim_artwork_index]['artist'],
                 'imageUrl':self._df_all_metadata.iloc[sim_artwork_index]['imageUrl'],
                 'sim_value':sim_matrix[:,sim_artwork_index][0]
                }, 
               ignore_index=True)

            #Save predicted artwork's code
            self._predicted_code_list.append(self._all_data_matrix[sim_artwork_index])

            #Save predicted artwork's index
            indexes_appended_list.append(sim_artwork_index)

        self._df_predicted_tour = self._df_predicted_tour.set_index('id')
        return self._df_predicted_tour
