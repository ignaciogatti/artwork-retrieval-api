# artwork-deep-learning-api

This an API to expose machine learning service for Arts.

It was developed using:
- Python 3.
- Flask
- Flaskplus
- Keras

## Artwork retrieval

![artwork-retrieval-motivation](https://github.com/ignaciogatti/art-deep-learning/blob/master/images/Artwork-retrieval.jpg)

Artwork-retrieval api define the logic to search similar artworks using deep-autoencoders (defined on Auotencoder-artwork notebook). Basically, first we encode each image of the dataset it using a pre-trained encoder, obtaining a code matrix. Then, given an artwork, we encode it with the same encoder. Then, we look foward the most similar codes of the matrix. Finally, we take the top ten artworks associated to these codes. Punctually, here we use a similarity measure that combines cosine distance and social influence. Basically, the idea is to adjust the cosine distance taking into account how far away are the artists in the influence social graph (This graph was built using DBpedia Ontology).

To summarize, the endpoint (called predict) recibe an artwork as input and return a list of similar artworks.
