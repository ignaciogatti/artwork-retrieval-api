from flask import request
from flask_restplus import Resource
from ..service.artwork_encode_service import predict
from ..utils.dto import ArtworkDto
from ..utils.parsers import file_upload
from ..utils.logger import write_cloud_logger
from ..utils.storage_utils import upload_blob
import os.path
from werkzeug.utils import secure_filename
import numpy as np


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
BASE_DIR = os.path.join(os.getcwd(),'static/img')

api = ArtworkDto.api

@api.route('/predict/')
class ArtworkCodeMatrix(Resource):

    def allowed_file(self, filename):
        return ('.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

    
    @api.doc('code_matrix_arwroks')
    @api.expect(file_upload)
    def post(self):
        """Encode artwork"""
        write_cloud_logger('Begin Post method')
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
            
            return {
                #pass image as str
                'sim_artwroks': predict(img_str)
                }
        
        return {
                'data':'',
                'message':'Something when wrong',
                'status':'error'
                }

