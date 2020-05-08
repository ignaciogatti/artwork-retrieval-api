from flask import request
from flask_restplus import Resource
from werkzeug.utils import secure_filename
import numpy as np
import os.path
from ..utils.dto import ArtworkDto
from ..utils.parsers import file_upload
from ..utils.logger import write_cloud_logger
from ..utils.storage_utils import upload_blob
from ..service.artwork_retrieval_service import Artwork_retrieval_service
from ..utils.similarity_measure import Cosine_similarity, Wasserstein_similarity
from ..utils.sort_utils import Social_influence_sort


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
BASE_DIR = os.path.join(os.getcwd(),'static/img')

api = ArtworkDto.api

@api.route('/social_graph/predict/')
class ArtworkCodeMatrix(Resource):


    def allowed_file(self, filename):
        return ('.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

    
    @api.doc('social_graph_artworks')
    @api.expect(file_upload)
    def post(self):
        """Encode artwork"""
        data = file_upload.parse_args()
        if data['image_file'] == "":
            return {
                    'data':'',
                    'message':'No file found',
                    'status':'error'
                    }
        photo = data['image_file']

        if photo and self.allowed_file(photo.filename):
            filename = secure_filename(photo.filename)
            img_str = photo.read()
            #Upload image to cloud storage
            upload_blob(filename, img_str, photo.content_type)

            #Define similarity measure
            sim_measure = Cosine_similarity()
            #sim_measure = Wasserstein_similarity()

            #Define sort algorithm
            sort_algorithm = Social_influence_sort()

            #Define Artwork retrieval service
            artwork_retrieval_service = Artwork_retrieval_service(sim_measure, sort_algorithm)
            
            return {
                #pass image as str
                'sim_artworks': artwork_retrieval_service.predict(img_str)
                }
        
        return {
                'data':'',
                'message':'Something when wrong',
                'status':'error'
                }

