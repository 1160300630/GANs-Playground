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
tanh = tf.tanh

# WGAN-GP 64 x 64
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
        img = tanh(dconv(y, 3, 5, 2))
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

# WGAN-GP 28 x 28
def generator_231(z, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    fc_lrelu = partial(fc, activation_fn=lrelu)
    dconv_relu = partial(dconv, activation_fn=relu)

    with tf.variable_scope('generator', reuse=reuse):
        y = bn(fc_lrelu(z, 1024))
        y = bn(fc_lrelu(y, 7 * 7 * dim * 2))
        y = tf.reshape(y, (-1, 7, 7, dim * 2))
        y = bn(dconv_relu(y, dim, 4, 2))
        img = tanh(dconv(y, 1, 4, 2))
        return img

def discriminator_WGAN_231(img, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_lrelu = partial(conv, activation_fn=lrelu)
    fc_lrelu = partial(fc, activation_fn=lrelu)

    with tf.variable_scope('discriminator', reuse=reuse):
        y = tf.reshape(img, [-1, 28, 28, 1])
        y = conv_lrelu(y, dim, 4, 2)
        y = bn(conv_lrelu(y, dim*2, 4, 2))
        y = fc_lrelu(y, 1024)
        logits = fc(y, 1)
        return logits

def generator_AC_GAN(z, t, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu, biases_initializer=None)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    dconv_tanh = partial(dconv, activation_fn=tanh, biases_initializer=None)

    with tf.variable_scope('generator', reuse=reuse):
        t = bn(fc_relu(t, 256))
        z = bn(fc_relu(z, 256))
        y = tf.concat((z, t), axis=1)
        y = bn(fc_relu(y, 1024))
        y = bn(fc_relu(y, 7 * 7 * dim * 2))
        y = tf.reshape(y, [-1, 7, 7, dim * 2])
        y = dconv_bn_relu(y, dim, 4, 2)
        img = tf.nn.tanh(dconv(y, 1, 4, 2))
        return img


def discriminator_AC_GAN(img, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_lrelu = partial(conv, normalizer_fn=None, activation_fn=lrelu, biases_initializer=None)
    fc_lrelu = partial(fc, normalizer_fn=None, activation_fn=lrelu)
    
    with tf.variable_scope('discriminator', reuse=reuse):
        y = tf.reshape(img, [-1, 28, 28, 1])
        y = lrelu(conv(y, dim, 4, 2))
        y = bn(conv_lrelu(y, dim*2, 4, 2))
        y = fc_lrelu(y, 1024)
        logits = fc(y, 1)
        return logits

def classifier_AC_GAN(img, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu)
    
    with tf.variable_scope('classifier', reuse=reuse):
        y = fc_bn_lrelu(img, 1024)
        y = fc_bn_lrelu(y, 512)
        logits = fc(y, 10)
        return logits