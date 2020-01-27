import tensorflow as tf
import numpy as np
import cv2 as cv
from operator import mul
from functools import reduce

class Classifier:
    _graph = None
    _input_height = 224
    _input_width = 224
    _input_mean = 0
    _input_std = 255
    _sess = None
    _input_operation = None
    _output_operation = None

    def __init__(self,
                 input_size,
                 graph_filename,
                 input_layer,
                 output_layer,
                 input_mean=0,
                 input_std=255):
        self._input_width, self._input_height = input_size
        self.load_graph(graph_filename)
        self._input_operation = self._graph.get_operation_by_name('import/{0}'.format(input_layer))
        self._output_operation = self._graph.get_operation_by_name('import/{0}'.format(output_layer))
        self._input_mean = input_mean
        self._input_std = input_std

    def __del__(self):
        if self._sess:
            self._sess.close()

    # return (index, top_k)
    def classify(self, image):
        if not self._sess:
            self.init_sess()
        result = self._sess.run(self._output_operation.outputs[0], {
            self._input_operation.outputs[0]: self.read_tensor_from_image_file(image)
        })
        result = np.squeeze(result).tolist()
        return result.index(max(result)), result

    def load_graph(self, model_file):
        self._graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with self._graph.as_default():
            tf.import_graph_def(graph_def)
        return self._graph

    def read_tensor_from_image_file(self, image_mat):
        resized = cv.resize(image_mat, (self._input_width, self._input_height))
        np_data = np.asarray(resized)
        np_data = np.expand_dims(np_data, axis=0)
        np_data = np.divide(np_data.astype('float'), self._input_std)
        # np_data = cv.normalize(np_data.astype('float'),
        #                        None,
        #                        1.0,
        #                        0,
        #                        cv.NORM_L1)
        return np_data

    def init_sess(self):
        if not self._sess:
            self._sess = tf.Session(graph=self._graph)
