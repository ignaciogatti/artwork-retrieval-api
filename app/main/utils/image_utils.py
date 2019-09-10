import cv2  # for image processing
import numpy as np

def get_image(filestr, img_Width=128, img_Height=128):
    #load image

    #convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    #image = cv2.imread(image_path)
    image = cv2.resize(image, (img_Width, img_Height), interpolation=cv2.INTER_CUBIC)
    #normalize image
    image_norm = image * (1./255)
    image_norm = np.expand_dims(image_norm, axis=0)
    
    return image_norm