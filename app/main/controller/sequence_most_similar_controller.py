from flask import request
from flask_restplus import Resource
from werkzeug.utils import secure_filename
import numpy as np
import os.path
from ..utils.dto import ArtworkDto
from ..utils.parsers import file_upload_list
from ..utils.logger import write_cloud_logger
from ..utils.storage_utils import upload_blob
from ..service.sequence_most_similar_service import Artwork_sequence_most_similar_service
from ..utils.similarity_measure import Cosine_similarity, Wasserstein_similarity
from ..utils.sort_utils import Naive_sort, Social_influence_sort
from ..utils.mongodb_utils import insert_user_image_key


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
BASE_DIR = os.path.join(os.getcwd(),'static/img')

api = ArtworkDto.api

@api.route('/sequence/mostsimilar/predict/')
class ArtworkSequenceMostSimilar(Resource):


    def allowed_file(self, filename):
        return ('.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

    
    @api.doc('sequence_most_similar_artworks')
    @api.expect(file_upload_list)
    def post(self):
        """Encode artwork"""
        data = file_upload_list.parse_args()
        
        window_images = []
        #Get window images
        image_one = data['image_file_one']
        image_two = data['image_file_two']
        image_three = data['image_file_three']

        #Get userId
        userId = data['userId']

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
        file_id = ''
        window_matrix_input = []
        for img in window_images:
            if img and self.allowed_file(img.filename):
                filename = secure_filename(img.filename)
                img_str = img.read()
                #Upload image to cloud storage
                upload_blob(filename, img_str, img.content_type)
                window_matrix_input.append(img_str)

                #Link image with the user
                if userId != None:
                    inserted_id = insert_user_image_key(userId, filename)
                    file_id += filename+'|'

            else:
                return {
                'data':'',
                'message':'Something when wrong',
                'status':'error'
                }


        #Define Artwork sequence most similar service
        artwork_sequence_most_similar_service = Artwork_sequence_most_similar_service()
        
        #Complete the file id name
        if userId != None:
            file_id += userId
        

        return {
            'file_id': str(file_id),
            #pass image as str
            'sim_artworks': artwork_sequence_most_similar_service.predict_tour(window_matrix_input)
            }
    

