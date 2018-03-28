from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim

from functools import partial

conv = partial(slim.conv2d, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
dconv = partial(slim.conv2d_transpose, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
fc = partial(ops.flatten_fully_connected, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
relu = tf.nn.relu
lrelu = partial(ops.leaky_relu, leak=0.2)
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
ln = slim.layer_norm

def generator(z, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope('generator', reuse=reuse):
        y = fc_bn_relu(z, 4 * 4 * dim * 8)
        y = tf.reshape(y, [-1, 4, 4, dim * 8])
        y = dconv_bn_relu(y, dim * 4, 5, 2)
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        y = dconv_bn_relu(y, dim * 1, 5, 2)
        img = tf.tanh(dconv(y, 3, 5, 2))
        return img

def generator_2(z, dim=64, reuse=False):

    with tf.variable_scope('generator', reuse=reuse):
        Wconv1 = tf.get_variable("Wconv1", shape=[5, 5, 256, dim*8])
        bconv1 = tf.get_variable("bconv1", shape=[256])
        Wconv2 = tf.get_variable("Wconv2", shape=[5, 5, 128, 256])
        bconv2 = tf.get_variable("bconv2", shape=[128])
        Wconv3 = tf.get_variable("Wconv3", shape=[5, 5, 64, 128])
        bconv3 = tf.get_variable("bconv3", shape=[64])
        Wconv4 = tf.get_variable("Wconv4", shape=[5, 5, 3, 64])
        bconv4 = tf.get_variable("bconv4", shape=[3])

        
        
        fc1 = tf.layers.dense(z, 4 * 4 * dim * 8, activation=tf.nn.relu)
        bn1 = tf.layers.batch_normalization(fc1, training=True)
        relu1 = tf.nn.relu(bn1)
        relu1_flatten = tf.reshape(relu1, [-1, 4, 4, dim*8])
        conv_transpose1 = tf.nn.conv2d_transpose(relu1_flatten, Wconv1, output_shape=[tf.shape(z)[0], 8, 8, dim * 4], strides=[1, 2, 2, 1]) + bconv1
        bn2 = tf.layers.batch_normalization(conv_transpose1, training=True)
        relu2 = tf.nn.relu(bn2)
        
        conv_transpose2 = tf.nn.conv2d_transpose(relu2, Wconv2, output_shape=[tf.shape(z)[0], 16, 16, dim * 2], strides=[1, 2, 2, 1]) + bconv2
        bn3 = tf.layers.batch_normalization(conv_transpose2, training=True)
        relu3 = tf.nn.relu(bn3)
        
        conv_transpose3 = tf.nn.conv2d_transpose(relu3, Wconv3, output_shape=[tf.shape(z)[0], 32, 32, dim * 1], strides=[1, 2, 2, 1]) + bconv3
        bn4 = tf.layers.batch_normalization(conv_transpose3, training=True)
        relu4 = tf.nn.relu(bn4)
        
        conv_transpose4 = tf.nn.conv2d_transpose(relu4, Wconv4, output_shape=[tf.shape(z)[0], 64, 64, 3], strides=[1, 2, 2, 1]) + bconv4                
        img = tf.tanh(conv_transpose4)
        return img

def discriminator(img, dim=64, reuse=True, training=True):
    conv_ln_lrelu = partial(conv, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope('discriminator', reuse=reuse):
        y = lrelu(conv(img, dim, 5, 2))
        y = conv_ln_lrelu(y, dim * 2, 5, 2)
        y = conv_ln_lrelu(y, dim * 4, 5, 2)
        y = conv_ln_lrelu(y, dim * 8, 5, 2)
        logit = fc(y, 1)
        return logit