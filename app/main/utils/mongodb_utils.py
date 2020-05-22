from pymongo import MongoClient
import os
import json


BASE_DIR = os.path.join(os.getcwd(),'static')

with open(os.path.join(BASE_DIR, 'mongodb_client.json')) as json_file:
            data_dict = json.loads(json_file.read())
#Connect with Atlas MongoDB
client = MongoClient(data_dict['client'])

#Get collection
db = client['test']
collection = db['user_image_upload']

def insert_user_image_key(userId, filename):
    data_to_insert={
        'userId' : userId,
        'filename' : filename
    }
    
    #insert into the collection
    x = collection.insert_one(data_to_insert)
    return x.inserted_id