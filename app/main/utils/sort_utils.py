import numpy as numpy
import json
import os.path
from abc import ABC, abstractclassmethod


MODEL_DIR = os.path.join(os.getcwd(), 'static/model')
INFLUENCE_GRAPH_PATH = os.path.join(MODEL_DIR, 'shortest_path_length.js')

class Base_sort(ABC):

    @abstractclassmethod
    def get_sorted_artworks(self, df):
        pass


class Naive_sort(Base_sort):

    def get_sorted_artworks(self, df):
        return  df.sort_values(by=['sim_distance'], ascending=False)


class Social_influence_sort(Base_sort):

    def __init__(self, artist_source):
        # Test load json file
        with open(INFLUENCE_GRAPH_PATH) as json_file:
            self.shortest_path_length = json.loads(json_file.read()) 
        self.artist_ocurrence = 0
        self.artist_source = artist_source


    def sim_influence(self, sim_distance, artist_target):

        if self.artist_source == artist_target:
            artist_decay = 2 ** artist_ocurrence
            artist_ocurrence += 1
            return sim_distance * (1./artist_decay)
        if artist_target in self.shortest_path_length[self.artist_source]:
            return sim_distance * (1./self.shortest_path_length[artist_source][artist_target])
        else:
            return sim_distance * (1./100)


    def get_sorted_artworks(self, df):
        self.artist_ocurrence = 0
        df['sim_influence'] = df.apply(
            lambda x: self.sim_influence(sim_distance=x['sim_distance'], 
            artist_target=x['artist']),axis=1 )
        return  df.sort_values(by=['sim_influence'], ascending=False)
