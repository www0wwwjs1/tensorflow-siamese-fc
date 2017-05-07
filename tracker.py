# Suofei ZHANG, 2017.

import os
import tensorflow as tf

import utils
from siamese_net import SiameseNet
from parameters import configParams

def getOpts():
    opts = {}
    print("config opts...")

    opts['scaleStep'] = 1.0375
    opts['scalePenalty'] = 0.9745
    opts['scaleLr'] = 0.59
    opts['responseUp'] = 16
    opts['windowing'] = 'cosine'
    opts['wInfluence'] = 0.176
    opts['exemplarSize'] = 127
    opts['instanceSize'] = 255
    opts['scoreSize'] = 17
    opts['totalStride'] = 8
    opts['contextAmount'] = 0.5
    opts['trainWeightDecay'] = 5e-04
    opts['stddev'] = 0.03

    opts['video'] = 'vot15_bag'
    opts['modelPath'] = './models/'
    opts['modelName'] = opts['modelPath']+"model_epoch30.ckpt"
    opts['summaryFile'] = './data_track/'+opts['video']+'_2'

    return opts

'''----------------------------------------main-----------------------------------------------------'''
def main(_):
    print('run tracker...')
    opts = getOpts()
    params = configParams()

    exemplarOp = tf.placeholder(tf.float32, [params['trainBatchSize'], opts['exemplarSize'], opts['exemplarSize'], 3])

    sn = SiameseNet()
    zFeatOp = sn.buildExemplarSubNetwork(exemplarOp, opts)

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, opts['modelName'])

    zFeat = sess.run(zFeatOp, feed_dict={exemplarOp: exemplar})
    print(zFeat)



if __name__=='__main__':
    tf.app.run()

    # saver = tf.train.import_meta_graph(opts['modelName']+'.meta')
    # graph = tf.get_default_graph()
    # weights = graph.get_tensor_by_name("weights:read:0")


