# artwork-deep-learning-api

This an API to expose machine learning service for Arts. It is deployed in [here](https://art-retrieval-api-234614.appspot.com/), using [Google App Engine](https://cloud.google.com/appengine/) (sometimes the link is broken because it is under development and have a limit of request). App Engine allows to deploy your app without worry to configure the server stuff.

![API-screenshot](https://github.com/ignaciogatti/art-deep-learning/blob/master/images/Swagger-description.png)

The API was developed using:
- Python 3.
- Tensorflow 2.1

To expose the server, we use:
- Flask
- Flaskplus
- Gunicorn


## Artwork retrieval

Our approach consists of a hybrid RS in which a user can submit an artwork of her interest, in the form of a digital image, and the RS will generate a list of artworks being related to the input image.  A relation between a given pair of images is established in terms of the visual contents of the images and their contexts. 
Basically, we aim at combining a **visual analysis** of the artwork (e.g., color and composition palette) with a **semantic analysis** of the artwork's context (e.g., stylistic influences among painters).

![artwork-retrieval-motivation](https://github.com/ignaciogatti/art-deep-learning/blob/master/images/Artwork-retrieval.jpg)

Internally, the RS relies on three building blocks in order to achieve its goal, as shown in Fig. First, we assume that a Deep Autoencoder is built, based on a predefined collection of artworks. Thus, when the user submits an image (*artwork<sub>target</sub>*), it gets codified using the **Deep Autoencoder**. Second, the encoded image is compared for similarity against a dataset of images, which have been encoded beforehand. Those images being most similar to the input image are selected and a ranking *S* is returned. Third, the selected images undergoes a filtering process based on a network of **influences** among the artworks' painters, which adjusts the values of the ranking. The final ranking *S* is returned to the user.

### Deep Autoencoder

The *visual content* analysis relies on **Deep Autoencoders**. This is a DNN technique that is used for representing images in a low dimensional space. In other words, the goal of **the Deep Autoencoder** is to achieve an n-dimensional representation of the image (Encode<sub>layer</sub>) that contains its main features (For further information you can check this [link](https://www.deeplearningbook.org/contents/autoencoders.html)). 
