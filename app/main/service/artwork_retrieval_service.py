import tensorflow as tf
from tensorflow.python.keras.models import load_model
import pandas as pd
import numpy as np
import cv2  # for image processing
import os.path
from os import listdir
from ..utils.logger import write_cloud_logger
from ..utils.image_utils import get_image
from .abstract_artwork_retrieval_service import Abstract_artwork_retrieval_service


MODEL_DIR = os.path.join(os.getcwd(), 'static/model')

#Wasserstein Auto-encoder
WASSERSTEIN_PATH = os.path.join(MODEL_DIR, 'wasserstein_encoder.h5')
WASSERSTEIN_MATRIX_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_w_encoded.npy' )

wasserstein_model = {
    'model_encoder' : WASSERSTEIN_PATH,
    'matrix' : WASSERSTEIN_MATRIX_FILE_NAME
}

#Denoisy Auto-encoder
MODEL_PATH = os.path.join(MODEL_DIR, 'denoisy_encoder.h5')
MATRIX_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encode.npy' )

denoisy_model = {
    'model_encoder' : MODEL_PATH,
    'matrix' : MATRIX_FILE_NAME
}

##Select model ## 
model = denoisy_model
#model = wasserstein_model



class Artwork_retrieval_service(Abstract_artwork_retrieval_service):

    def __init__(self, sim_measure, sort_algorithm):
        super().__init__(sim_measure, sort_algorithm)
        self.name = "Artwork_retrieval_generic_service"        
        #load matrix model
        self.artwork_code_matrix = np.load( model['matrix'] )
        



    def predict(self, filestr):
        
        image_norm = get_image(filestr)
        #load Auto-encoder model
        autoencoder_model = load_model(model['model_encoder'])
         #Get code of the input image       
        code = autoencoder_model.predict(image_norm).reshape((-1,))
        #os.remove(image_path)
        
        return self.get_sim_artworks(code, self.artwork_code_matrix)


