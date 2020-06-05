import pandas as pd
import numpy as np
import cv2  # for image processing
import os.path
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from os import listdir
from ..utils.logger import write_cloud_logger
from ..utils.image_utils import get_image
from ..utils_sequence.sequence_generation_rnn import Sequence_generator_rnn


MODEL_DIR = os.path.join(os.getcwd(), 'static/model')

#Denoisy Auto-encoder
MODEL_PATH = os.path.join(MODEL_DIR, 'denoisy_encoder.h5')
MATRIX_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encode.npy' )

ALL_METADATA_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encoded_with_url.csv' )


model = {
    'model_encoder' : MODEL_PATH,
    'metadata' : ALL_METADATA_FILE_NAME,
    'matrix' : MATRIX_FILE_NAME
}


class Artwork_sequence_rnn_service:

    def __init__(self):
        self.name = "Artwork_sequence_rnn_service"
        #Load all metadata
        self._all_metadata = pd.read_csv(model['metadata'])
        #load Auto-encoder model
        self._autoencoder_model = load_model(model['model_encoder'])
        #load matrix model
        self._artwork_code_matrix = np.load( model['matrix'] )
        #load sequence RNN mdoel
        self._sequence_rnn_model = Sequence_generator_rnn(self._all_metadata, self._artwork_code_matrix)


    def predict_tour(self, window_images):

        #Define window for sequence RNN input
        img_codes =[]
        for img_str in window_images:

            image_norm = get_image(img_str)
            #Get the code for the input image
            code = self._autoencoder_model.predict(image_norm).reshape((-1,))
            img_codes.append(code)  
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
        
        return result
