import numpy as numpy
import pickle
import os.path
from abc import ABC, abstractclassmethod
import unicodedata

MODEL_DIR = os.path.join(os.getcwd(), 'static/model')
INFLUENCE_GRAPH_PATH = os.path.join(MODEL_DIR, 'artist_shortest_path_length.pkl')

class Base_sort(ABC):

    @abstractclassmethod
    def get_sorted_artworks(self, df):
        pass


class Naive_sort(Base_sort):

    def get_sorted_artworks(self, df):
        return  df.sort_values(by=['sim_distance'], ascending=False)


class Social_influence_sort(Base_sort):

    def __init__(self, artist_source=''):
        # Test load pickle file
        with open(INFLUENCE_GRAPH_PATH, 'rb') as pickle_model:
            self.shortest_path_length = pickle.load(pickle_model)
        print( len(self.shortest_path_length.keys()) ) 
        self.artist_ocurrence = 0
        self.artist_source = self.normalize_title(artist_source)


    def normalize_title(self, title):
        return unicodedata.normalize('NFKD', title.lower()).encode('ASCII', 'ignore').decode('utf8')


    def sim_influence(self, sim_distance, artist_target):
        
        artist_target = self.normalize_title(artist_target)
        if self.artist_source == artist_target:
            artist_decay = 2 ** self.artist_ocurrence
            self.artist_ocurrence += 1
            return sim_distance + sim_distance * (1./artist_decay)
        
        #Sort artist name for policy to search in dictionary
        sorted_artists = sorted([artist_target, self.artist_source])
        #Check if the artist is in the graph
        if not sorted_artists[0] in set(self.shortest_path_length.keys()):
            return sim_distance + sim_distance * (1./100)
        if sorted_artists[1] in self.shortest_path_length[sorted_artists[0]]:
            return sim_distance + sim_distance * (1./self.shortest_path_length[sorted_artists[0]][sorted_artists[1]])
        else:
            return sim_distance + sim_distance * (1./100)


    def get_artist_source(self, df):
        artist_keys = self.shortest_path_length.keys()
        df_sorted = df[['artist', 'sim_distance']].sort_values(by=['sim_distance'], ascending=False)
        for index, row in df_sorted.iterrows():
            artist_name = row['artist']
            artist_name = self.normalize_title(artist_name)
            if artist_name in artist_keys:
                return artist_name
        #Default value
        return 'pablo picasso'

    def get_sorted_artworks(self, df):
        #It is supposed that it will be the one with the closest cosine distance
        self.artist_source = self.get_artist_source(df)
        #print(self.artist_source)
        self.artist_ocurrence = 0
        df['sim_influence'] = df.apply(
            lambda x: self.sim_influence(sim_distance=x['sim_distance'], 
            artist_target=x['artist']),axis=1 )
        return  df.sort_values(by=['sim_influence'], ascending=False)
