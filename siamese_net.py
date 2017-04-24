# Suofei ZHANG, 2017.

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

MOVING_AVERAGE_DECAY = 0        #only siamese-net in matlab uses 0 here, other projects with tensorflow all use 0.999 here, from some more documents, I think 0.999 is more probable here, since tensorflow uses a equation as 1-decay for this parameter
UPDATE_OPS_COLLECTION = 'resnet_update_ops'

# def inference(_instance):
#     # input of network z
#     exemplar = tf.placeholder('float32', [None, 127, 127, 3])
#     # input of network x
#     a_feat = tf.placeholder('float32', [None, 6, 6, 256])
#     instance = tf.placeholder('float32', [None, 255, 255, 3])
#     # self.score = tf.placeholder('float32', [None, 17, 17, 1])
#
# def train(params):
#     return

def buildNetwork(exemplar, instance, isTraining):
    with tf.variable_scope('siamese') as scope:
        aFeat = buildBranch(exemplar, isTraining)
        scope.reuse_variables()
        score = buildBranch(instance, isTraining)

        # the conv2d op in tf is used to implement xcorr directly, from theory, the implementation of conv2d is correlation. However, it is necessary to transpose the weights tensor to a input tensor
        # different scales are tackled with slicing the data. Now only 3 scales are considered, but in training, more samples in a batch is also tackled by the same mechanism. Hence more slices is to be implemented here!!
    with tf.variable_scope('scorr'):
        print("Building xcorr...")
        groupConv = lambda i, k: tf.nn.conv2d(i, k, strides=[1, 1, 1, 1], padding='VALID')
        aFeat = tf.transpose(aFeat, perm=[1, 2, 3, 0])
        shapeAFeat = aFeat.get_shape()
        if int(shapeAFeat[-1]) > 1:
            aFeats = tf.split(axis=3, num_or_size_splits=shapeAFeat[-1], value=aFeat)

        scores = tf.split(axis=3, num_or_size_splits=score.get_shape()[-1], value=score)
        scores = [groupConv(i, k) for i, k in zip(scores, aFeats)]

        score = tf.concat(axis=3, values=scores)

        # shapeAFeat = aFeat.get_shape()
        # aFeat0 = tf.slice(aFeat, [0, 0, 0, 0], [shapeAFeat[0], shapeAFeat[1], shapeAFeat[2], 1])
        # aFeat1 = tf.slice(aFeat, [0, 0, 0, 1], [shapeAFeat[0], shapeAFeat[1], shapeAFeat[2], 1])
        # aFeat2 = tf.slice(aFeat, [0, 0, 0, 2], [shapeAFeat[0], shapeAFeat[1], shapeAFeat[2], 1])
        #
        # shapeScore = score.get_shape()
        # score0 = tf.slice(score, [0, 0, 0, 0], [1, shapeScore[1], shapeScore[2], shapeScore[3]])
        # score1 = tf.slice(score, [1, 0, 0, 0], [1, shapeScore[1], shapeScore[2], shapeScore[3]])
        # score2 = tf.slice(score, [2, 0, 0, 0], [1, shapeScore[1], shapeScore[2], shapeScore[3]])
        #
        # score0 = tf.nn.conv2d(score0, aFeat0, strides=[1, 1, 1, 1])
        # score1 = tf.nn.conv2d(score1, aFeat0, strides=[1, 1, 1, 1])
        # score2 = tf.nn.conv2d(score2, aFeat0, strides=[1, 1, 1, 1])
        #
        # score = tf.concat([score0, score1, score2], 0)

    with tf.variable_scope('adjust'):
        print("Building adjust...")
        weights = tf.get_variable('weights', [1, 1, 1, 1], initializer=tf.constant_initializer(value=0.001, dtype=tf.float32))
        biases = tf.get_variable('biases', [1, ], initializer=tf.constant_initializer(value=0, dtype=tf.float32))
        score = tf.nn.conv2d(score, weights, strides=[1, 1, 1, 1])
        score = tf.add(score, biases)

    return score

def buildBranch(inputs, isTraining):
    print("Building Siamese branches...")

    with tf.variable_scope('scala1'):
        print("Building conv1, bn1, relu1, pooling1...")
        # outputs = conv1(inputs, 3, 96, 11, 2)
        outputs = conv(inputs, 96, 11, 2, 1)
        outputs = batchNormalization(outputs, isTraining)
        outputs = tf.nn.relu(outputs)
        outputs = maxPool(outputs, 3, 2)

    with tf.variable_scope('scala2'):
        print("Building conv2, bn2, relu2, pooling2...")
        # outputs = conv2(outputs, 48, 256, 5, 1)
        outputs = conv(outputs, 256, 5, 1, 2)
        outputs = batchNormalization(outputs, isTraining)
        outputs = tf.nn.relu(outputs)
        outputs = maxPool(outputs, 3, 2)

    with tf.variable_scope('scala3'):
        print("Building conv3, bn3, relu3...")
        # outputs = conv1(outputs, 256, 384, 3, 1)
        outputs = conv(outputs, 384, 3, 1, 1)
        outputs = batchNormalization(outputs, isTraining)
        outputs = tf.nn.relu(outputs)

    with tf.variable_scope('scala4'):
        print("Building conv4, bn4, relu4...")
        # outputs = conv2(outputs, 192, 384, 3, 1)
        outputs = conv(outputs, 384, 3, 1, 2)
        outputs = batchNormalization(outputs, isTraining)
        outputs = tf.nn.relu(outputs)

    with tf.variable_scope('scala5'):
        print("Building conv5...")
        # outputs = conv2(outputs, 192, 256, 3, 1)
        outputs = conv(outputs, 256, 3, 1, 2)

    return outputs

def conv(inputs, filters, size, stride, groups):
    channels = int(inputs.get_shape()[-1])
    groupConv = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride, stride, 1], padding='VALID')

    weights = tf.get_variable('weights', shape=[size, size, channels/groups, filters], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    biases = tf.get_variable('biases', [filters,], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))

    if groups == 1:
        conv = groupConv(inputs, weights)
    else:
        inputGroups = tf.split(axis=3, num_or_size_splits=groups, value=inputs)
        weightsGroups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
        convGroups = [groupConv(i, k) for i, k in zip(inputGroups, weightsGroups)]

        conv = tf.concat(axis=3, values=convGroups)

    conv = tf.add(conv, biases)
    print('Layer Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d, Groups = %d' % (size, size, stride, filters, channels, groups))

    return conv

# deprecated
def conv1(inputs, channels, filters, size, stride):
    # initializations include trancated norm distribution method and xavier method, the matlab version exploits an improved xavier method.
    # However I didn't find it in tf, so xavier is used here, if not work, something may need change here!!
    weights = tf.get_variable('weights', [size, size, channels, filters], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    biases = tf.get_variable('biases', [filters,], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))

    conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='VALID')
    conv = tf.add(conv, biases)
    print('Layer Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (size, size, stride, filters, channels))

    return conv

# deprecated
def conv2(inputs, channels, filters, size, stride):
    inputShape = inputs.get_shape()

    inputs0 = tf.slice(inputs, [0, 0, 0, 0], [inputShape[0], inputShape[1], inputShape[2], channels])
    inputs1 = tf.slice(inputs, [0, 0, 0, channels], [inputShape[0], inputShape[1], inputShape[2], channels])

    weights0 = tf.get_variable('weights0', [size, size, channels, filters/2], initializer=tf.contrib.layers.xavier_initializer_conv2d(), dtype=tf.float32)
    weights1 = tf.get_variable('weights1', [size, size, channels, filters/2], initializer=tf.contrib.layers.xavier_initializer_conv2d(), dtype=tf.float32)

    conv0 = tf.nn.conv2d(inputs0, weights0, strides=[1, stride, stride, 1])
    conv1 = tf.nn.conv2d(inputs1, weights1, strides=[1, stride, stride, 1])
    conv = tf.concat([conv0, conv1], 3)

    biases = tf.get_variable('biases', [filters, ], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))
    conv = tf.add(conv, biases)
    print('Layer Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (size, size, stride, filters, channels))

    return conv

def batchNormalization(inputs, isTraining):
    return tf.contrib.layers.batch_norm(inputs, center=True, scale=True, is_training=isTraining)
# def batchNormalization(inputs, isTraining):
#     xShape = inputs.get_shape()
#     paramsShape = xShape[-1:]
#     axis = list(range(len(xShape)-1))
#
#     beta = tf.get_variable('beta', paramsShape, initializer=tf.constant_initializer(value=0, dtype=tf.float32))
#     gamma = tf.get_variable('gamma', paramsShape, initializer=tf.constant_initializer(value=1, dtype=tf.float32))
#     movingMean = tf.get_variable('moving_mean', paramsShape, initializer=tf.constant_initializer(value=0, dtype=tf.float32), trainable=False)
#     movingVariance = tf.get_variable('moving_variance', paramsShape, initializer=tf.constant_initializer(value=1, dtype=tf.float32), trainable=False)
#
#     mean, variance = tf.nn.moments(inputs, axis)
#     updateMovingMean = moving_averages.assign_moving_average(movingMean, mean, MOVING_AVERAGE_DECAY)
#     updateMovingVariance = moving_averages.assign_moving_average(movingVariance, variance, MOVING_AVERAGE_DECAY)
#     tf.add_to_collection(UPDATE_OPS_COLLECTION, updateMovingMean)
#     tf.add_to_collection(UPDATE_OPS_COLLECTION, updateMovingVariance)
#
#     mean, variance = control_flow_ops.cond(isTraining, lambda: (mean, variance), lambda: (movingMean, movingVariance))
#
#     bn = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 0.0001)
#     print('Layer type = batch_norm')
#
#     return bn

def maxPool(inputs, kSize, _stride):
    return tf.nn.max_pool(inputs, ksize=[1, kSize, kSize, 1], strides=[1, _stride, _stride, 1], padding='VALID')





