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


## Artwork Recommender

Our approach consists of a hybrid RS in which a user can submit an artwork of her interest, in the form of a digital image, and the RS will generate a list of artworks being related to the input image.  A relation between a given pair of images is established in terms of the visual contents of the images and their contexts. 
Basically, we aim at combining a **visual analysis** of the artwork (e.g., color and composition palette) with a **semantic analysis** of the artwork's context (e.g., stylistic influences among painters).

<p align="center">
  <img src="https://github.com/ignaciogatti/art-deep-learning/blob/master/images/Artwork-retrieval.jpg" width="500" height="400">
</p>

Internally, the RS relies on three building blocks in order to achieve its goal, as shown in Fig. First, we assume that a Deep Autoencoder is built, based on a predefined collection of artworks. Thus, when the user submits an image (*artwork<sub>target</sub>*), it gets codified using the **Deep Autoencoder**. Second, the encoded image is compared for similarity against a dataset of images, which have been encoded beforehand. Those images being most similar to the input image are selected and a ranking *S* is returned. Third, the selected images undergoes a filtering process based on a network of **influences** among the artworks' painters, which adjusts the values of the ranking. The final ranking *S* is returned to the user.

### Deep Autoencoder

The *visual content* analysis relies on **Deep Autoencoders**. This is a DNN technique that is used for representing images in a low dimensional space. In other words, the goal of **the Deep Autoencoder** is to achieve an n-dimensional representation of the image (Encode<sub>layer</sub>) that contains its main features (For further information you can check this [link](https://www.deeplearningbook.org/contents/autoencoders.html)). 

<p align="center">
  <img src="https://github.com/ignaciogatti/art-deep-learning/blob/master/images/Autoencoder-description.jpg" width="500" height="400">
</p>

### Influence Social Graph

For the \textit{context based} part, we rely on semantic knowledge about the artworks. Following ideas from \cite{wang2009using} and \cite{Semeraro2012folkrecsys}, we enrich the model with an ontology as a network that relates different artworks. Specifically, we consider *influence* relationships among artists, as presented in the [DBpedia Ontology](https://wiki.dbpedia.org/services-resources/ontology). This relationship defines an **influence social graph** among artists as illustrated in figure below.

<p align="center">
  <img src="https://github.com/ignaciogatti/art-deep-learning/blob/master/images/influence-graph-example.jpg" width="500" height="400">
</p>


In this context, we compute a coefficient that reflects the influence that a given artist has on others. The rationale for this coefficient is to keep close those artworks whose artists are related in the **influence social graph**. Let us suppose that *x, y* are nodes of the graph representing two artists, then the **influence** coefficient is defined as:  

![img_eq](http://latex.codecogs.com/svg.latex?inf%28x%2Cy%29%3D%5Cfrac%7B1%7D%7Bshortest%5C_path%28x%2Cy%29%7D)

For example, in the arts domain it is well-known that Dominique Ingres (1780-1867) was influenced by Rafael Sanzio (1483-1520). Likewise, Pierre Renoir (1841-1919) was influenced by Ingres’ work. Thus, it is feasible to state that the artworks of these three artists are related, and they should appear together in the set of artworks proposed by the RS. In [prior work](http://sedici.unlp.edu.ar/handle/10915/73027), we had results showing that an **influence social graph** can enhance the performance of a multi-domain RS, as well as in cold start scenarios.

### Recommendation

When a user asks for an artwork recommendation, we assume that the encoder model, the database of artworks and the influence social graph for them have already been computed. On this basis, the process consists of three steps. 
In the first step,  the user selects an artwork *artwork<sub>t</sub>* of her interest and uploads it to the RS. The RS computes the code of this artwork using the **Deep Autoencoder** model. With this representation, the second step is to look for other similar vectors in the database of R<sub>codes</sub> using a **Cosine similarity** : 

![img_eq](http://latex.codecogs.com/svg.latex?cos_%7Bsim%7D%28x%2Cy%29%3D%5Cfrac%7Bx%5Ccdot+y%7D%7B%7C%7Cx%7C%7C%2A%7C%7Cy%7C%7C%7D)

This step generates a ranking of similar artworks $S$.

In the third step, the similarity measure is adjusted by taking into account the influence relationships. To do this, the user should also enter the artist’s name for her artwork *artist<sub>target</sub>*: 

![img_eq](http://latex.codecogs.com/svg.latex?dist%28a_i%29%3Dcos_%7Bsim%7D%28artwork_%7Bt%7D%2Ca_i%29)

Then, the similarity value of each artwork *a<sub>i</sub>* in *S* is weighted by applying the **influence coefficient**. This coefficient is computed based on the distance between *artist<sub>t</sub>* and the artist of each artwork *artist<sub>i<sub>*. 

![img_eq](http://latex.codecogs.com/svg.latex?inf%28x%2Cy%29%3D%5Cfrac%7B1%7D%7Bshortest%5C_path%28x%2Cy%29%7D)

With the adjusted values, the RS sorts *S* by moving the most similar artworks to the top of the ranking. Since we are interested in having diversity in the recommended set, sets of artworks of artworks that belong only to a few artists should be avoided. Thus, we use another coefficient that punishes artworks if the artist has another artworks in the recommended set. Let us suppose that we traverse *S* in order. 

![img_eq](http://latex.codecogs.com/svg.latex?p%28a_i%29%3D%5Cfrac%7B1%7D%7B2%5E%7Bprevious%5C_occurrences%28artist_i%29%7D%7D)

For each artwork *a<sub>i</sub>* the occurrences of the artist *artist<sub>i</sub>* are counted, and then **punish coefficient** is applied.

At last, the recommended set is an artwork ranking that is sorted first by *visual content* similarity with *artwork<sub>t</sub>*, and then sorted again according to the \textit{influence relationships} between *artist<sub>t</sub>* and others artists. This means that the RS first looks for those artworks having a similar composition and color palette, and subsequently tries to keep close those artworks being related by culture. This proccess is summirezed in the following equation:

![img_eq](http://latex.codecogs.com/svg.latex?semantic_%7Bdist%7D%28a_i%29%3Ddist%28a_i%29+%2Bdist%28a_i%29%2Ainf%28artist_%7Bt%7D%2Cartist_i%29%2Ap%28a_i%29)

For example, if the user select a Monte's artwork, the *visual content* analysis will select artworks with similar composition. However, most of these artworks belongs to artists that have nothing in common with Renoir. After the *influence artist* analysis the ranking is reordered and those artworks whose artists are related to Monet appear in the top. 
