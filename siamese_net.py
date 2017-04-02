# Copyright (c) <2017> <SUOFEI ZHANG>. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

class SiameseNet:
    gpuId = '0'

    def __init__(self, _gpuId):   #argvs=[]
        self.gpuId = _gpuId
        self.buildNetwork()

    # def argv_parser(self, argvs):
    #     for i in range(1, len(argvs), 2):
    #         if argvs[i] == '-gpu' : self.gpuId = argvs[i+1]

    def buildNetwork(self):
        print "Building Siamese Network..."

        # input of network z
        self.exemplar = tf.placeholder('float32', [None, 127, 127, 3])
        # input of network x
        self.a_feat = tf.placeholder('float32', [None, 6, 6, 256])
        self.instance = tf.placeholder('float32', [None, 255, 255, 3])
        # self.score = tf.placeholder('float32', [None, 17, 17, 1])

        self.a_conv1 = self.convLayer('a_conv1', self.exemplar, 3, 96, 11, 2)
        self.a_bn1 =

    def convLayer(self, _name, inputs, channels, filters, size, stride):
        weights = tf.get_variable('weights', [size, size, channels, filters], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', [filters,], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))

        conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], name = _name+'conv')
        conv_biased = tf.add(conv, biases, _name)
        print '    Layer '+_name+': Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (size, size, stride, filters, channels)

        return conv_biased

    def batchNormalization(self):



