# Suofei ZHANG, 2017.

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

MOVING_AVERAGE_DECAY = 0        #只用siames-net这里用0，其他的tensorflow实现这里都用0.999左右的数字。看了一些资料，感觉这里应该用0.999，因为tensorflow的公式是个1-decay的公式，实验的时候要试一下
UPDATE_OPS_COLLECTION = 'resnet_update_ops'

def inference(_instance):
    # input of network z
    exemplar = tf.placeholder('float32', [None, 127, 127, 3])
    # input of network x
    a_feat = tf.placeholder('float32', [None, 6, 6, 256])
    instance = tf.placeholder('float32', [None, 255, 255, 3])
    # self.score = tf.placeholder('float32', [None, 17, 17, 1])

def train(params):
    return

def buildBranch(inputs, params):
    print "Building Siamese branches..."

    with tf.variable_scope('scala1'):
        print "Building conv1, bn1, pooling1..."
        conv1 = conv(inputs, 3, 96, 11, 2)
        bn1 = batchNormalization(conv1, params['bn_is_training'])

def conv(inputs, channels, filters, size, stride):
    # xavier初始化和截断正态分布初始化，matlab版本用的是一种改进的xavier初始化，如果训练不行，这里可能要调
    weights = tf.get_variable('weights', [size, size, channels, filters], initializer=tf.contrib.layers.xavier_initializer_conv2d(), dtype=tf.float32)
    biases = tf.get_variable('biases', [filters,], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))

    conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1])
    conv_biased = tf.add(conv, biases)
    print 'Layer Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (size, size, stride, filters, channels)

    return conv_biased

def batchNormalization(inputs, isTraining):
    xShape = inputs.get_shape()
    paramsShape = xShape[-1:]
    axis = list(range(len(xShape)-1))

    beta = tf.get_variable('beta', paramsShape, initializer=tf.zeros_initializer)
    gamma = tf.get_variable('gamma', paramsShape, initializer=tf.ones_initializer)
    movingMean = tf.get_variable('moving_mean', paramsShape, initializer=tf.zeros_initializer, trainable=False)
    movingVariance = tf.get_variable('moving_variance', paramsShape, initializer=tf.ones_initializer, trainable=False)

    mean, variance = tf.nn.moments(inputs, axis)
    updateMovingMean = moving_averages.assign_moving_average(movingMean, mean, MOVING_AVERAGE_DECAY)
    updateMovingVariance = moving_averages.assign_moving_average(movingVariance, variance, MOVING_AVERAGE_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, updateMovingMean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, updateMovingVariance)

    mean, variance = control_flow_ops.cond(isTraining, lambda: (mean, variance), lambda: (movingMean, movingVariance))

    bn = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 0.0001)
    print 'Layer type = batch_norm'

    return bn





