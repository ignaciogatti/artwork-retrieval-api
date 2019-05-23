import tensorflow as tf
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import cv2  # for image processing
import os.path
from os import listdir
from ..utils.logger import write_cloud_logger


MODEL_DIR = os.path.join(os.getcwd(), 'static/model')
MODEL_PATH = os.path.join(MODEL_DIR, 'inception_impressionism.h5')


def get_image(filestr, img_Width=256, img_Height=256):
    #load image

    #convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    #image = cv2.imread(image_path)
    image = cv2.resize(image, (img_Width, img_Height), interpolation=cv2.INTER_CUBIC)
    #expand image (necessary for preprocess function input)
    image = np.expand_dims(image, axis=0)
    #normalize image
    image_norm = preprocess_input(image)

    return image_norm


def predict(filestr):
    
    image_norm = get_image(filestr)
    model = load_model(MODEL_PATH)
    result = model.predict(image_norm).reshape((-1,))
    #os.remove(image_path)
    return str(result[0])
