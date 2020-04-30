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

# Embedding matrix
EMB_MATRIX_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_embedding.npy' )

denoisy_embedding_model = {
    'model_encoder' : MODEL_PATH,
    'matrix' : MATRIX_FILE_NAME,
    'emb_matrix' :EMB_MATRIX_FILE_NAME
}

##Select model ## 
model = denoisy_embedding_model



class Code_Embedding_Artwork_retrieval_service(Abstract_artwork_retrieval_service):

    def __init__(self, sim_measure, sort_algorithm):
        super().__init__(sim_measure, sort_algorithm)
        self.name = "Code_Embedding_Artwork_retrieval_service"        
        #load matrix model
        self.artwork_code_matrix = np.load( model['matrix'] )
        self.embedding_code_matrix = np.load( model['emb_matrix'] )
        self.code_emb_matrix = np.hstack((self.artwork_code_matrix, self.embedding_code_matrix))
        



    def predict(self, filestr):
        
        image_norm = get_image(filestr)
        #load Auto-encoder model
        autoencoder_model = load_model(model['model_encoder'])
         #Get code of the input image       
        code = autoencoder_model.predict(image_norm).reshape((-1,))
        #os.remove(image_path)

        #Find artwork with the most similar code
        sim_matrix = self.sim_measure.get_similarity_measure_matrix(code, self.artwork_code_matrix)
        artwork_index = np.argsort(sim_matrix)[0,-1]

        embedding_code = self.embedding_code_matrix[artwork_index,:]

        code_emb = np.hstack((code, embedding_code))
        print(code_emb.shape)
        
        return self.get_sim_artworks(code_emb, self.code_emb_matrix)


