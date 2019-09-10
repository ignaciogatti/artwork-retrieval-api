import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os.path

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(os.getcwd(),'static/model')
JSON_CREDENTIALS = os.path.join( BASE_DIR, 'artwork-retrieval.json' )
METADATA_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encoded.csv' )
MATRIX_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encode.npy' )


def get_sim_arworks(image_id):

    #Load data
    df_artworks = pd.read_csv( METADATA_FILE_NAME )
    artwork_code_matrix = np.load( MATRIX_FILE_NAME )

    #get similarity matrix for image_id
    sim_matrix = cosine_similarity(artwork_code_matrix[int(image_id)].reshape((1,-1)), artwork_code_matrix)
    
    #get top-n most similar
    index_sorted = np.argsort(sim_matrix)
    top_n = index_sorted[0][-26:-1]
    top_n_matrix = np.take(a=sim_matrix, indices=top_n)
    
    df_top_n = df_artworks.iloc[top_n]
    df_top_n['sim_distance'] = top_n_matrix
    
    df_top_ten = df_top_n.sort_values(by=['sim_distance'], ascending=False)
    #df_top_ten = df_top_ten.head(10)
    return list(df_top_ten['filename'].values)
