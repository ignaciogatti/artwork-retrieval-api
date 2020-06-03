from flask_restplus import Api
from flask import Blueprint

from .controller.artwork_retrieval_controller import api as encode_ns
from .controller.cb_artwork_retrieval_controller import api as cb_encode_ns
from .controller.code_emb_artwork_retrieval_controller import api as code_emb_encode_ns
from .controller.social_graph_artwork_retrieval_controller import api as social_graph_encode_ns

from .controller.sequence_rnn_controller import api as sequence_rnn_ns
from .controller.sequence_most_similar_controller import api as sequence_most_similar_ns
from .controller.sequence_artist_rnn_controller import api as sequence_artist_rnn_ns

from .controller.impressionism_classifier_controller import api as impressionism_classifier_ns


main = Blueprint('api', __name__)

api = Api(main,
          title='ARTWORK RETRIEVAL API WITH FLASK RESTPLUS AND JWT',
          version='1.0',
          description='An artwrok retrieval api developed with flask restplus'
          )


api.add_namespace(encode_ns, path='/artwork')
api.add_namespace(impressionism_classifier_ns, path='/artwork')
api.add_namespace(cb_encode_ns, path='/artwork')
api.add_namespace(sequence_rnn_ns, path='/artwork')
api.add_namespace(code_emb_encode_ns, path='/artwork')
api.add_namespace(social_graph_encode_ns, path='/artwork')
api.add_namespace(sequence_most_similar_ns, path='/artwork')
api.add_namespace(sequence_artist_rnn_ns, path='/artwork')
