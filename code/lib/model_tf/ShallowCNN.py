# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: Sound Source Localization
# @File: ShallowCNN.py
# @Time: 2021/11/29/10:59
# @Software: PyCharm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Dropout, Conv2D, AveragePooling2D, \
    BatchNormalization
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K


# need these for ShallowConvNet
def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


def denoising(l, name, embed=True, softmax=True):
    """
    Feature Denoising, Fig 4 & 5.
    """
    with tf.variable_scope(name):
        f = non_local_op(l, embed=embed, softmax=softmax)
        f = Conv2D(int(l.shape[1]), (1, 1), strides=1)(f)
        f = l + f
    
    return f


def non_local_op(l, embed, softmax):
    """
    Feature Denoising, Sec 4.2 & Fig 5.
    Args:
        embed (bool): whether to use embedding on theta & phi
        softmax (bool): whether to use gaussian (softmax) version or the dot-product version.
    """
    n_in, H, W = l.shape.as_list()[1:]
    if embed:
        theta = Conv2D(n_in, (1, 1), strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))(l)
        phi = Conv2D(n_in, (1, 1), strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))(l)
        g = Conv2D(n_in, (1, 1), strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))(l)
    else:
        theta, phi, g = l, l, l
    
    theta = tf.transpose(theta, [0, 2, 3, 1])
    phi = tf.transpose(phi, [0, 2, 3, 1])
    g = tf.transpose(g, [0, 2, 3, 1])
    theta_flat = tf.reshape(theta,
                            [-1, theta.shape.as_list()[1] * theta.shape.as_list()[2], theta.shape.as_list()[-1]])
    phi_flat = tf.reshape(phi, [-1, phi.shape.as_list()[1] * phi.shape.as_list()[2], phi.shape.as_list()[-1]])
    g_flat = tf.reshape(g, [-1, g.shape.as_list()[1] * g.shape.as_list()[2], g.shape.as_list()[-1]])
    f = tf.matmul(theta_flat, tf.transpose(phi_flat, [0, 2, 1]))
    if softmax:
        # f = f / tf.sqrt(n_in)
        f = tf.nn.softmax(f)
        fg = tf.matmul(f, g_flat)
        fg = tf.transpose(tf.reshape(fg, [-1, *g.shape.as_list()[1:]]), [0, 3, 1, 2])
    else:
        f = f / tf.cast(H * W, tf.float32)
        fg = tf.matmul(f, g_flat)
        fg = tf.transpose(tf.reshape(fg, [-1, *g.shape.as_list()[1:]]), [0, 3, 1, 2])
    
    return fg


def ShallowConvNet(num_classes, Chans=64, SamplePoints=128, dropoutRate=0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.

    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25

    Note that this implementation has not been verified by the original
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations.
    """
    
    # start the model
    input_main = Input((1, Chans, SamplePoints))
    block1 = Conv2D(40, (1, 13),
                    input_shape=(1, Chans, SamplePoints),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    # if is_denoising:
    #     block1 = Lambda(lambda t: denoising(t, name='denosing_1', embed=True, softmax=True))(block1)
    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1 = Activation(log)(block1)
    if dropoutRate is not None:
        block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(num_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


if __name__ == '__main__':
    
    print('Hello World!')
