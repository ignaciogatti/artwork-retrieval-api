import tensorflow as tf
from keras.models import load_model
import pandas as pd
import numpy as np
import cv2  # for image processing
import os.path
from os import listdir
from ..utils.logger import write_cloud_logger
from ..utils.image_utils import get_image



MODEL_DIR = os.path.join(os.getcwd(), 'static/model')
MODEL_PATH = os.path.join(MODEL_DIR, 'denoisy_encoder.h5')
METADATA_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encoded_with_url.csv' )
MATRIX_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encode.npy' )


class Artwork_retrieval_service:

    def __init__(self, sim_measure, sort_algorithm):
        self.name = "Artwork_retrieval_generic_service"
        #load data
        self.df_artworks = pd.read_csv( METADATA_FILE_NAME )
        self.artwork_code_matrix = np.load( MATRIX_FILE_NAME )
        self.sim_measure = sim_measure
        self.sort_algorithm = sort_algorithm


    def get_sim_artworks(self, code):
        
        #get similarity matrix for image_id
        sim_matrix = self.sim_measure.get_similarity_measure_matrix(code, self.artwork_code_matrix)
        
        #get top-n most similar
        index_sorted = np.argsort(sim_matrix)
        top_n = index_sorted[0][-100:]
        top_n_matrix = np.take(a=sim_matrix, indices=top_n)
        
        df_top_n = self.df_artworks.iloc[top_n]
        df_top_n['sim_distance'] = top_n_matrix
        

        df_top_ten = self.sort_algorithm.get_sorted_artworks(df_top_n)
        df_top_ten = df_top_ten.dropna(subset=['imageUrl'])
        df_top_ten = df_top_ten.head(25)
        top_ten = df_top_ten[['title', 'artist', 'imageUrl']].transpose().to_dict()
        values = list(top_ten.values())
        
        result = []
        for i in range(len(top_ten)):
            values[i]['id'] = list(top_ten.keys())[i]
            result.append(values[i])

        return result


    def predict(self, filestr):
        
        image_norm = get_image(filestr)
        model = load_model(MODEL_PATH)
        code = model.predict(image_norm).reshape((-1,))
        #os.remove(image_path)
        return self.get_sim_artworks(code)


