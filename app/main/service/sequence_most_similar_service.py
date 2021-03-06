import pandas as pd
import numpy as np
import cv2  # for image processing
import os.path
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from os import listdir
from ..utils.logger import write_cloud_logger
from ..utils.image_utils import get_image
from ..utils_sequence.sequence_generation_most_similar import Sequence_generator_based_previous_most_similar


MODEL_DIR = os.path.join(os.getcwd(), 'static/model')

#Denoisy Auto-encoder
MODEL_PATH = os.path.join(MODEL_DIR, 'denoisy_encoder.h5')
MATRIX_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encode.npy' )

denoisy_model = {
    'model_encoder' : MODEL_PATH,
    'matrix' : MATRIX_FILE_NAME
}


class Artwork_sequence_most_similar_service:

    def __init__(self):
        self.name = "Artwork_sequence_most_similar_service"
        #load sequence Most Similar mdoel
        self._sequence_most_similar_model = Sequence_generator_based_previous_most_similar()
        #load Auto-encoder model
        self._autoencoder_model = load_model(denoisy_model['model_encoder'])


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
        self._sequence_most_similar_model.set_tour(x_tour_matrix)
        df_tour_predicted = self._sequence_most_similar_model.predict_tour()

        #Drop artworks without valid url
        df_tour_predicted = df_tour_predicted.dropna(subset=['imageUrl'])

        top_ten = df_tour_predicted[['title', 'artist', 'imageUrl']].transpose().to_dict()
        values = list(top_ten.values())
        
        result = []
        #15 it is length for the sequence recomendation
        for i in range(min(15, len(top_ten))):
            values[i]['id'] = list(top_ten.keys())[i]
            result.append(values[i])

        return result

        #os.remove(image_path)
        
        return result
