import pandas as pd
import numpy as np
import cv2  # for image processing
import os.path
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from os import listdir
from ..utils.logger import write_cloud_logger
from ..utils.image_utils import get_image
from ..utils_sequence.sequence_generation_artist_rnn import Sequence_generator_artist_rnn
from ..utils.similarity_measure import Cosine_similarity


MODEL_DIR = os.path.join(os.getcwd(), 'static/model')

#Denoisy Auto-encoder
MODEL_PATH = os.path.join(MODEL_DIR, 'denoisy_encoder.h5')
MATRIX_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encode.npy' )

ALL_METADATA_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encoded_with_url.csv' )

# Embedding matrix
EMB_MATRIX_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_embedding.npy' )

# Artist code matrix
ARITST_CODE_MATRIX_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_artist_code_matrix.npy' )

model = {
    'model_encoder' : MODEL_PATH,
    'metadata' : ALL_METADATA_FILE_NAME,
    'matrix' : MATRIX_FILE_NAME,
    'emb_matrix' : EMB_MATRIX_FILE_NAME,
    'artist_code_matrix' : ARITST_CODE_MATRIX_FILE_NAME
}


class Artwork_sequence_artist_rnn_service:

    def __init__(self):
        self.name = "Artwork_sequence_artist_rnn_service"
        #load Auto-encoder model
        self._autoencoder_model = load_model(model['model_encoder'])

        #Create a similarity measure object
        self.sim_measure = Cosine_similarity()

        #Load all metadata
        self._all_metadata = pd.read_csv(model['metadata'])
    
        #load matrix model
        self.artwork_code_matrix = np.load( model['matrix'] )
        #load embedding matrix model
        self.embedding_code_matrix = np.load( model['emb_matrix'] )
        #Combine code with embeddings
        self.code_emb_matrix = np.hstack((self.artwork_code_matrix, self.embedding_code_matrix))

        #Reduce artist code matrix
        self.artist_code_matrix = np.load( model['artist_code_matrix'] )
        self.artist_code_emb_matrix = np.hstack((self.code_emb_matrix, self.artist_code_matrix.reshape((-1, 1))))

        #load sequence RNN mdoel
        self._sequence_rnn_model = Sequence_generator_artist_rnn(self._all_metadata, self.artist_code_emb_matrix)



    def predict_tour(self, window_images):

        #Define window for sequence RNN input
        img_codes =[]
        for img_str in window_images:

            image_norm = get_image(img_str)
            #Get the code for the input image
            code = self._autoencoder_model.predict(image_norm).reshape((-1,))

            #Find artwork with the most similar code
            sim_matrix = self.sim_measure.get_similarity_measure_matrix(code, self.artwork_code_matrix)
            artwork_index = np.argsort(sim_matrix)[0,-1]

            #Append most similar embedding and artist code
            artist_code_emb = self.artist_code_emb_matrix[artwork_index,:]            

            img_codes.append(artist_code_emb)  
        x_tour_matrix = np.stack(img_codes)

        print(x_tour_matrix.shape)

       
        #Predict tour
        self._sequence_rnn_model.set_tour(x_tour_matrix)
        df_tour_predicted = self._sequence_rnn_model.predict_tour()

        #Drop artworks without valid url
        df_tour_predicted = df_tour_predicted.dropna(subset=['imageUrl'])

        top_ten = df_tour_predicted[['title', 'artist', 'imageUrl']].transpose().to_dict()
        values = list(top_ten.values())
        
        result = []
        for i in range(len(top_ten)):
            values[i]['id'] = list(top_ten.keys())[i]
            result.append(values[i])

        return result

        #os.remove(image_path)
        
