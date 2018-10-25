import ops
from ops import conv_sn, dconv_sn, fc_sn
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

# Self-Attention GAN
def generator_SA_GAN(z, t, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu, biases_initializer=None)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    dconv_tanh = partial(dconv, activation_fn=tanh, biases_initializer=None)

    with tf.variable_scope('generator', reuse=reuse):
        t = bn(lrelu(fc_sn(t, dim*4)))
        z = bn(lrelu(fc_sn(z, dim*4)))
        y = tf.concat((z, t), axis=1)
        y = bn(lrelu(fc_sn(y, 7 * 7 * dim * 2)))
        y = tf.reshape(y, [-1, 7, 7, dim * 2])
        y = bn(lrelu(dconv_sn(y, dim, 4, 2)))
        y = attention(y, dim)
        img = tf.nn.tanh(dconv_sn(y, 1, 4, 2))
        return img

def discriminator_SA_GAN(img, dim=64, reuse=True, training=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        y = tf.reshape(img, [-1, 28, 28, 1])
        y = lrelu(conv_sn(y, dim, 4, 2))
        y = attention(y, dim)
        y = lrelu(conv_sn(y, dim*2, 4, 2))
        feature = attention(y, dim*2)
        y = lrelu(fc_sn(feature, 1024))
        feature = lrelu(fc_sn(y, 1024))
        logits = fc_sn(y, 1)
    return logits, feature

def classifier_SA_GAN(feature, dim=64, reuse=True, training=True):
    with tf.variable_scope('classifier', reuse=reuse):
        y = lrelu(fc_sn(feature, 1024))
        y = lrelu(fc_sn(y, 1024))
        logits = fc_sn(y, 10)
        return logits

def attention(inputs, dim, scope=None):
    with tf.variable_scope(scope, 'attention', [inputs]):
        f = conv_sn(inputs=inputs, dim=(dim//8), kernel_size=1, stride=1)
        g = conv_sn(inputs=inputs, dim=(dim//8), kernel_size=1, stride=1)
        h = conv_sn(inputs=inputs, dim=dim, kernel_size=1, stride=1)

        s = tf.matmul(g, f, transpose_b=True)
        attention_shape = tf.shape(s)
        #s = tf.reshape(s, shape=[attention_shape[0], -1, attention_shape[-1]]) # [batch_size, N. dim]
        s = tf.reshape(s, shape=[attention_shape[0], -1, attention_shape[-1]]) # [batch_size, N, dim]

        beta = tf.nn.softmax(s, dim=1) # attention map
        beta = tf.reshape(beta, shape=attention_shape)
        o = tf.matmul(beta, h)

        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        outputs = gamma * o + inputs

        return  output

def conv_sn(inputs, dim, kernel_size, stride, scope=None):
    weight_init = tf.contrib.layers.xavier_initializer()
    
    with tf.variable_scope(scope, 'conv_sn', [inputs]):
        w = tf.get_variable("kernel", shape=[kernel_size, kernel_size, inputs.get_shape()[-1], dim], initializer=weight_init)
        outputs = tf.nn.conv2d(input=inputs, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding='VALID')
        bias = tf.get_variable("bias", [dim], initializer=tf.constant_initializer(0.0))
        outptus = tf.nn.bias_add(outputs, bias)

        return outputs

def dconv_sn(inputs, dim, kernel_size, stride, scope=None):
    weight_init = tf.contrib.layers.xavier_initializer()
    
    with tf.variable_scope(scope, 'dconv_sn', [inputs]):
        inputs_shape = inputs.get_shape().as_list()
        outputs_shape = [tf.shape(inputs)[0], inputs_shape[1]*stride, inputs_shape[2]*stride, dim]
        w = tf.get_variable('kernel', shape=[kernel_size, kernel_size, dim, inputs.get_shape()[-1]], initializer=weight_init)
        outputs = tf.nn.conv2d_transpose(inputs, filter=spectral_norm(w), output_shape=outputs_shape, strides=[1, stride, stride, 1], padding='SAME')
        bias = tf.get_variable('bias', [dim], initializer=tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, bias)
        return outputs

def fc_sn(inputs, dim, scope=None):
    weight_init = tf.contrib.layers.xavier_initializer()
    
    with tf.variable_scope(scope, 'fc_sn', [inputs]):
        inputs = tf.layers.flatten(inputs)
        shape = inputs.get_shape().as_list()
        channels = shape[-1]

        w = tf.get_variable('kernel', [channels, dim], tf.float32, initializer=weight_init)
        bias = tf.get_variable('bias', [dim], initializer=tf.constant_initializer(0.0))
        outputs = tf.matmul(inputs, spectral_norm(w)) + bias
        return outputs


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable('u', [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)