from flask import request
from flask_restplus import Resource
from werkzeug.utils import secure_filename
import numpy as np
import os.path
from ..utils.dto import ArtworkDto
from ..utils.parsers import file_upload_list
from ..utils.logger import write_cloud_logger
from ..utils.storage_utils import upload_blob
from ..service.sequence_artist_rnn_service import Artwork_sequence_artist_rnn_service
from ..utils.similarity_measure import Cosine_similarity, Wasserstein_similarity
from ..utils.sort_utils import Naive_sort, Social_influence_sort


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
BASE_DIR = os.path.join(os.getcwd(),'static/img')

api = ArtworkDto.api

@api.route('/sequence/artistrnn/predict/')
class ArtworkSequenceArtistRNN(Resource):


    def allowed_file(self, filename):
        return ('.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

    
    @api.doc('sequence_artist_rnn_artworks')
    @api.expect(file_upload_list)
    def post(self):
        """Encode artwork"""
        data = file_upload_list.parse_args()
        
        window_images = []
        #Get window images
        image_one = data['image_file_one']
        image_two = data['image_file_two']
        image_three = data['image_file_three']

        window_images.append(image_one)
        window_images.append(image_two)
        window_images.append(image_three)

        #Check no empty file
        for img in window_images:
            if img == "":
                return {
                        'data':'',
                        'message':'No file found',
                        'status':'error'
                        }
        
        #Create a window for input model
        window_matrix_input = []
        for img in window_images:
            if img and self.allowed_file(img.filename):
                filename = secure_filename(img.filename)
                img_str = img.read()
                #Upload image to cloud storage
                upload_blob(filename, img_str, img.content_type)
                window_matrix_input.append(img_str)
            else:
                return {
                'data':'',
                'message':'Something when wrong',
                'status':'error'
                }


        #Define Artwork sequence RNN service
        artwork_sequence_artist_rnn_service = Artwork_sequence_artist_rnn_service()
        

        return {
            'file_id': '',
            #pass image as str
            'sim_artworks': artwork_sequence_artist_rnn_service.predict_tour(window_matrix_input)
            }
    

