import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

WEIGHT_INIT_STDDEV = 0.1


class Decoder(object):

    def __init__(self, model_pre_path):
        self.weight_vars = []
        self.model_pre_path = model_pre_path

        with tf.variable_scope('decoder'):

            self.weight_vars.append(self._create_variables(64, 64, 3, scope='conv2_1'))
            self.weight_vars.append(self._create_variables(64, 32, 3, scope='conv2_2'))
            self.weight_vars.append(self._create_variables(32, 16, 3, scope='conv2_3'))
            self.weight_vars.append(self._create_variables(16, 1 , 3, scope='conv2_4'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):

        if self.model_pre_path:
            reader = pywrap_tensorflow.NewCheckpointReader(self.model_pre_path)
            with tf.variable_scope(scope):
                kernel = tf.Variable(reader.get_tensor('decoder/' + scope + '/kernel'), name='kernel')
                bias = tf.Variable(reader.get_tensor('decoder/' + scope + '/bias'), name='bias')
        else:
            with tf.variable_scope(scope):
                shape = [kernel_size, kernel_size, input_filters, output_filters]
                kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
                bias = tf.Variable(tf.zeros([output_filters]), name='bias')
        return (kernel, bias)

    def decode(self, image):
        final_layer_idx  = len(self.weight_vars) - 1

        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]

            if i == final_layer_idx:
                out = conv2d(out, kernel, bias, use_relu=False)
            else:
                out = conv2d(out, kernel, bias)
            # print('decoder ', i)
            # print('decoder out:', out.shape)
        return out


def conv2d(x, kernel, bias, use_relu=True):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)

    if use_relu:
        out = tf.nn.relu(out)

    return out

