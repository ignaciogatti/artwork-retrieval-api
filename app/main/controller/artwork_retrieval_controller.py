from flask import request
from flask_restplus import Resource
from ..service.artwork_retrieval_service import get_sim_arworks
from ..utils.dto import ArtworkDto

api = ArtworkDto.api

@api.route('/<image_id>')
@api.param('image_id', 'The artwork identifier')
class ArtworkSimList(Resource):
    @api.doc('list_of_similar_arwroks')
    def get(self, image_id):
        """List similar artworks"""
        return get_sim_arworks(image_id)
