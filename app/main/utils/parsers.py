# parsers.py
from werkzeug.datastructures import FileStorage
from flask_restplus import reqparse

file_upload = reqparse.RequestParser()
file_upload.add_argument('image_file',  
                         type=FileStorage, 
                         location='files', 
                         required=True, 
                         help='Image file')

file_upload_list = reqparse.RequestParser()
file_upload_list.add_argument('image_file_list',  
                         type=FileStorage, 
                         location='files', 
                         required=True,
                         action='append', 
                         help='Image files list')