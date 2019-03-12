import tensorflow as tf
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2  # for image processing
import os.path


MODEL_DIR = os.path.join(os.getcwd(), 'static/model')
MODEL_PATH = os.path.join(MODEL_DIR, 'denoisy_encoder.h5')
METADATA_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encoded.csv' )
MATRIX_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encode.npy' )


def get_image(image_path, img_Width=128, img_Height=128):
    #load image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_Width, img_Height), interpolation=cv2.INTER_CUBIC)
    #normalize image
    image_norm = image * (1./255)
    image_norm = np.expand_dims(image_norm, axis=0)
    
    return image_norm


def get_sim_artworks(code):
    
    #load data
    df_artworks = pd.read_csv( METADATA_FILE_NAME )
    artwork_code_matrix = np.load( MATRIX_FILE_NAME )

    #get similarity matrix for image_id
    sim_matrix = cosine_similarity(code.reshape((1,-1)), artwork_code_matrix)
    
    #get top-n most similar
    index_sorted = np.argsort(sim_matrix)
    top_n = index_sorted[0][-11:-1]
    top_n_matrix = np.take(a=sim_matrix, indices=top_n)
    
    df_top_n = df_artworks.iloc[top_n]
    df_top_n['sim_distance'] = top_n_matrix
    
    df_top_ten = df_top_n.sort_values(by=['sim_distance'], ascending=False)
    #df_top_ten = df_top_ten.head(10)
    return list(df_top_ten['filename'].values)


def predict(image_path):
    
    image_norm = get_image(image_path)
    model = load_model(MODEL_PATH)
    code = model.predict(image_norm).reshape((-1,))
    return get_sim_artworks(code)
