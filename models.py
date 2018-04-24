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
def generator_WGAN(z, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    dconv_tanh = partial(dconv, activation_fn=tanh, biases_initializer=None)

    with tf.variable_scope('generator', reuse=reuse):
        y = fc_bn_relu(z, 1024)
        y = fc_bn_relu(y, 7 * 7 * dim * 2)
        y = tf.reshape(y, [-1, 7, 7, dim * 2])
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        img = tanh(dconv(y, 1, 5, 2))
        return img

def discriminator_WGAN(img, dim=64, reuse=True, training=True):
    conv_ln_lrelu = partial(conv, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)
    fc_ln_lrelu = partial(fc, normalizer_fn=ln, activation_fn=lrelu)

    with tf.variable_scope('discriminator', reuse=reuse):
        y = tf.reshape(img, [-1, 28, 28, 1])
        y = lrelu(conv(y, 1, 5, 2))
        y = conv_ln_lrelu(y, dim, 5, 2)
        y = fc_ln_lrelu(y, 1024)
        logits = fc(y, 1)
        return logits

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

def discriminator_DCGAN_231(img, dim=64, reuse=True, training=True):
    fc_lrelu = partial(fc, activation_fn=lrelu)
    conv_lrelu = partial(conv, activation_fn=lrelu)
    pool = partial(tf.nn.max_pool, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    with tf.variable_scope('discriminator', reuse=reuse):
        y = tf.reshape(img, [-1, 28, 28, 1])
        y = pool(conv_lrelu(y, 32, 5, 1))
        y = pool(conv_lrelu(y, dim, 5, 1))
        y = fc_lrelu(y, 1024)
        logits = fc(y, 1)
        return logits

# Conditional WGAN-GP 28 x 28
def generator_C_GAN(z, t, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    dconv_tanh = partial(dconv, activation_fn=tanh, biases_initializer=None)

    with tf.variable_scope('generator', reuse=reuse):
        h = fc_bn_relu(t, z.shape[1])
        y = tf.concat((z, h), axis=0)
        y = fc_bn_relu(y, 1024)
        y = fc_bn_relu(y, 7 * 7 * dim * 2)
        y = tf.reshape(y, [-1, 7, 7, dim * 2])
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        img = tanh(dconv(y, 1, 5, 2))
        return img


def discriminator_C_WGAN(img, t, dim=64, reuse=True, training=True):
    conv_ln_lrelu = partial(conv, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)
    fc_ln_lrelu = partial(fc, normalizer_fn=ln, activation_fn=lrelu)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_sigmoid = partial(tf, normalizer_fn=bn, activation_fn=tf.sigmoid)

    with tf.variable_scope('discriminator', reuse=reuse):
        h = fc_bn_sigmoid(t, 14)
        h = dconv_bn_relu(h, 1, 5, 2)
        y = tf.reshape(img, [-1, 28, 28, 1])
        y = tf.concat((img, h), axis=0)
        y = lrelu(conv(y, 1, 5, 2))
        y = conv_ln_lrelu(y, dim, 4, 2)
        y = fc_ln_lrelu(y, 1024)
        logits = fc(y, 1)
        return logits

def generator_C_231(z, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    fc_lrelu = partial(fc, activation_fn=lrelu)
    dconv_relu = partial(dconv, activation_fn=relu)

    with tf.variable_scope('generator', reuse=reuse):
        h = fc_bn_relu(t, z.shape[1])
        y = tf.concat((z, h), axis=0)
        y = bn(fc_lrelu(z, 1024))
        y = bn(fc_lrelu(y, 7 * 7 * dim * 2))
        y = tf.resize(y, (-1, 7, 7, dim * 2))
        y = bn(dconv_relu(y, dim, 4, 2))
        y = tanh(dconv(y, 1, 4, 2))

def discriminator_C_WGAN_231(img, dim=64, reuse=True, traning=True):
    conv_lrelu = partial(conv, activation_fn=lrelu)
    fc_lrelu = partial(fc, activation_fn=lrelu)

    with tf.variable_scope('discriminator', reuse=reuse):
        h = fc_bn_sigmoid(t, 14)
        h = dconv_bn_relu(h, 1, 5, 2)
        y = tf.reshape(img, [-1, 28, 28, 1])
        y = tf.concat((img, h), axis=0)
        y = conv_lrelu(y, dim, 4, 2)
        y = bn(conv_lrelu(y, dim*2, 4, 2))
        y = fc_lrelu(y, 1024)
        logits = fc(y, 1)
        return logits

def discriminator_C_DCGAN_231(img, dim=64, reuse=True, training=True):
    fc_lrelu = partial(fc, activation_fn=lrelu)
    conv_lrelu = partial(conv, activation_fn=lrelu)
    pool = partial(tf.nn.max_pool, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    with tf.variable_scope('discriminator', reuse=reuse):
        h = fc_bn_sigmoid(t, 14)
        h = dconv_bn_relu(h, 1, 5, 2)
        y = tf.reshape(img, [-1, 28, 28, 1])
        y = tf.concat((img, h), axis=0)
        y = pool(conv_lrelu(y, 32, 5, 1))
        y = pool(conv_lrelu(y, dim, 5, 1))
        y = fc_lrelu(y, 1024)
        logits = fc(y, 1)
        return logits
