from flask import Flask
from flask import request
from flask import jsonify

from src.demo import *

GLOVE_DIMENSION = 100
CHUNK_SIZE = 5


class APIServer:
    def __init__(self):
        sess = tf.Session()
        checkpoint = tf.train.latest_checkpoint('../models')
        saver = tf.train.import_meta_graph(checkpoint + '.meta')
        saver.restore(sess, checkpoint)
        sess.run(tf.global_variables_initializer())
        self.sess = sess
        self.W1 = tf.get_default_graph().get_tensor_by_name("W1:0")
        self.b1 = tf.get_default_graph().get_tensor_by_name("b1:0")

        self.W2 = tf.get_default_graph().get_tensor_by_name("W2:0")
        self.b2 = tf.get_default_graph().get_tensor_by_name("b2:0")


app = Flask(__name__)
api_server = APIServer()

glove_vectors = load_word_vectors("../data/glove.6B.100d.txt", GLOVE_DIMENSION)
dictionary = set([word.strip() for word in open('../data//words.txt', 'r')])


@app.route("/prediction", methods=['POST'])
def get_prediction():
    req_data = request.get_json()
    input = req_data['data']
    mbti_type = find_prediction(input, dictionary, CHUNK_SIZE, api_server.sess, glove_vectors, api_server.W1,
                                api_server.b1, api_server.W2, api_server.b2)
    return jsonify({'prediction': mbti_type})


if __name__ == '__main__':
    app.run(host='localhost', port=3000)
