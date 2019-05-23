from flask_restplus import Api
from flask import Blueprint

from .controller.artwork_retrieval_controller import api as artwork_ns
from .controller.artwork_encode_controller import api as encode_ns
from .controller.impressionism_classifier_controller import api as impressionism_classifier_ns

main = Blueprint('api', __name__)

api = Api(main,
          title='ARTWORK RETRIEVAL API WITH FLASK RESTPLUS AND JWT',
          version='1.0',
          description='An artwrok retrieval api developed with flask restplus'
          )

api.add_namespace(artwork_ns, path='/artwork')
api.add_namespace(encode_ns, path='/artwork')
api.add_namespace(impressionism_classifier_ns, path='/artwork')