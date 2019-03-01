from flask_restplus import Namespace, fields

class ArtworkDto:
    api = Namespace('artwork', description='artwork retrieval operations')