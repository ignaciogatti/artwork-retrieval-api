from sklearn.metrics.pairwise import cosine_similarity
import numpy as numpy
from abc import ABC, abstractclassmethod
from scipy.stats import wasserstein_distance

class Base_similarity(ABC):

    @abstractclassmethod
    def get_similarity_measure_matrix(self, code, artwork_code_matrix):
        pass


class Cosine_similarity(Base_similarity):

    def get_similarity_measure_matrix(self, code, artwork_code_matrix):
        return cosine_similarity(code.reshape((1,-1)), artwork_code_matrix)


class Wasserstein_similarity(Base_similarity):

    def get_similarity_measure_matrix(self, code, artwork_code_matrix):
        for i in range(artwork_code_matrix.shape[0]):
            sim_list.append(wasserstein_distance(code_image.reshape((-1,)), artwork_code_matrix[i]))
    
        sim_matrix = np.array(sim_list)
        return sim_matrix.reshape((1,-1))