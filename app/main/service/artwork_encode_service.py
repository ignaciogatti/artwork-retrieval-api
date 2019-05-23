import tensorflow as tf
from keras.models import load_model
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import cv2  # for image processing
import os.path
from os import listdir
from ..utils.logger import write_cloud_logger
#from ..utils.graph_utils import get_edge_dict, sim_influence


MODEL_DIR = os.path.join(os.getcwd(), 'static/model')
MODEL_PATH = os.path.join(MODEL_DIR, 'denoisy_encoder.h5')
METADATA_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encoded_with_url.csv' )
MATRIX_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encode.npy' )
SOCIAL_GRAPH_FILE_NAME = os.path.join( MODEL_DIR, 'artist-influences-edges.csv' )


def get_image(filestr, img_Width=128, img_Height=128):
    #load image

    #convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    #image = cv2.imread(image_path)
    image = cv2.resize(image, (img_Width, img_Height), interpolation=cv2.INTER_CUBIC)
    #normalize image
    image_norm = image * (1./255)
    image_norm = np.expand_dims(image_norm, axis=0)
    
    return image_norm

#TODO Re-order taking account artist influence
def get_sim_artworks(code):
    
    for f in listdir(MODEL_DIR):
        write_cloud_logger(f)
    #load data
    df_artworks = pd.read_csv( METADATA_FILE_NAME )
    artwork_code_matrix = np.load( MATRIX_FILE_NAME )

    '''
    #Create influence social graph
    df_edges = pd.read_csv(SOCIAL_GRAPH_FILE_NAME)
    artist_dict = get_edge_dict(df=df_edges, col_to_index='Artist', col_to_split='Influence', col_to_clean='Influence')
    g_artist = nx.from_dict_of_lists(artist_dict)
    nx.set_edge_attributes(g_artist, 'red', 'color')
    nx.set_node_attributes(g_artist, 'artist', 'type')
    length = dict(nx.all_pairs_shortest_path_length(g_artist))
    '''

    #get similarity matrix for image_id
    sim_matrix = cosine_similarity(code.reshape((1,-1)), artwork_code_matrix)
    
    #get top-n most similar
    index_sorted = np.argsort(sim_matrix)
    top_n = index_sorted[0][-100:]
    top_n_matrix = np.take(a=sim_matrix, indices=top_n)
    
    df_top_n = df_artworks.iloc[top_n]
    df_top_n['sim_distance'] = top_n_matrix
    
    '''
    #re-order taking account artist influence
    artist_source = df_top_n.iloc[-1]['artist']
    df_top_n['sim_influence'] = df_top_n.apply(
    lambda x: sim_influence(sim_distance=x['sim_distance'], artist_source=artist_source, artist_target=x['artist'], length),
    axis=1 )
    '''

    df_top_ten = df_top_n.sort_values(by=['sim_distance'], ascending=False)
    df_top_ten = df_top_ten.dropna(subset=['imageUrl'])
    df_top_ten = df_top_ten.head(10)
    top_ten = df_top_ten[['title', 'artist', 'imageUrl']].transpose().to_dict()
    values = list(top_ten.values())
    
    result = []
    for i in range(len(top_ten)):
        values[i]['id'] = list(top_ten.keys())[i]
        result.append(values[i])

    return result


def predict(filestr):
    
    image_norm = get_image(filestr)
    model = load_model(MODEL_PATH)
    code = model.predict(image_norm).reshape((-1,))
    #os.remove(image_path)
    return get_sim_artworks(code)
