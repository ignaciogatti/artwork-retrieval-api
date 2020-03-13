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

#Denoisy Auto-encoder
MODEL_PATH = os.path.join(MODEL_DIR, 'denoisy_encoder.h5')
MATRIX_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encode.npy' )

denoisy_model = {
    'model_encoder' : MODEL_PATH,
    'matrix' : MATRIX_FILE_NAME
}



#TF-IDF model
TFIDF_MATRIX_FILE_NAME = os.path.join( MODEL_DIR, 'tfidf_matrix.npy' )

tfidf_model = {
    'matrix' : TFIDF_MATRIX_FILE_NAME
}



class CB_Artwork_retrieval_service(Abstract_artwork_retrieval_service):

    def __init__(self, sim_measure, sort_algorithm):
        super().__init__(sim_measure, sort_algorithm)
        self.name = "Artwork_retrieval_generic_service"
        #load code matrix model
        self.artwork_code_matrix = np.load( denoisy_model['matrix'] )

        #load tfidf matrix model
        self.artwork_tfidf_matrix = np.load( tfidf_model['matrix'], allow_pickle = True )
        #Because it is a sparse matrix and it was saved into an array
        self.artwork_tfidf_matrix = self.artwork_tfidf_matrix.reshape((-1))[0]


    def predict(self, filestr):

        image_norm = get_image(filestr)
        #load Auto-encoder model
        autoencoder_model = load_model(denoisy_model['model_encoder'])
        code = autoencoder_model.predict(image_norm).reshape((-1,))
        #Find artwork with the most similar code
        sim_matrix = self.sim_measure.get_similarity_measure_matrix(code, self.artwork_code_matrix)
        artwork_index = np.argsort(sim_matrix)[0,-1]

        tfidf_code = self.artwork_tfidf_matrix[artwork_index,:]
        #os.remove(image_path)
        
        return self.get_sim_artworks(tfidf_code, self.artwork_tfidf_matrix)
