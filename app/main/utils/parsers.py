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
file_upload_list.add_argument('image_file_one',  
                         type=FileStorage, 
                         location='files', 
                         required=True,
                         help='Image files one')

file_upload_list.add_argument('image_file_two',  
                         type=FileStorage, 
                         location='files', 
                         required=True,
                         help='Image files two')

file_upload_list.add_argument('image_file_three',  
                         type=FileStorage, 
                         location='files', 
                         required=True,
                         help='Image files three')