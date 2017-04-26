# Suofei ZHANG, 2017.

import numpy as np
import tensorflow as tf
import random
import siamese_net as sn
from parameters import configParams
import utils

def getOpts():
    opts = {}
    print("config opts...")

    opts['validation'] = 0.1
    opts['exemplarSize'] = 127
    opts['instanceSize'] = 255-2*8
    opts['lossRPos'] = 16
    opts['lossRNeg'] = 0
    opts['labelWeight'] = 'balanced'
    opts['numPairs'] = 53200
    opts['frameRange'] = 100
    opts['trainNumEpochs'] = 50
    opts['trainLr'] = np.logspace(-2, -5, opts['trainNumEpochs'])
    opts['trainWeightDecay'] = 5e-04
    opts['augTranslate'] = True
    opts['augMaxTranslate'] = 4
    opts['augStretch'] = True
    opts['augMaxStretch'] = 0.05
    opts['augColor'] = True
    opts['augGrayscale'] = 0
    opts['randomSeed'] = 0

    return opts

def getEig(mat):
    d, v = np.linalg.eig(mat)
    idx = np.argsort(d)
    d.sort()
    d = np.diag(d)
    v = -v;
    v = v[:, idx]

    return d, v

def loadStats(path):
    imgStats = utils.loadImageStats(path)

    if 'z' not in imgStats:
        print("to implement...")
        return
    else:
        rgbMeanZ = np.reshape(imgStats['z']['rgbMean'], [1, 1, 3])
        rgbMeanX = np.reshape(imgStats['x']['rgbMean'], [1, 1, 3])
        d, v = getEig(imgStats['z']['rgbCovariance'])
        rgbVarZ = 0.1*np.dot(np.sqrt(d), v.T)
        d, v = getEig(imgStats['x']['rgbCovariance'])
        rgbVarX = 0.1*np.dot(np.sqrt(d), v.T)
        return rgbMeanZ, rgbVarZ, rgbMeanX, rgbVarX

def chooseValSet(imdb, opts):
    TRAIN_SET = 1
    VAL_SET = 2

    sizeDataset = len(imdb.id)
    sizeVal = round(opts['validation']*sizeDataset)
    sizeTrain = sizeDataset-sizeVal
    imdb.set = np.zeros([sizeDataset], dtype='uint8')
    imdb.set[:sizeTrain] = TRAIN_SET
    imdb.set[sizeTrain:] = VAL_SET

    imdbInd = {}
    imdbInd['id'] = range(0, opts['numPairs'])
    imdbInd['imageSet'] = np.zeros([opts['numPairs']], dtype='uint8')
    nPairsTrain = round(opts['numPairs']*(1-opts['validation']))
    imdbInd['imageSet'][:nPairsTrain] = TRAIN_SET
    imdbInd['imageSet'][nPairsTrain:] = VAL_SET

    return imdb, imdbInd


def main(_):
    params = configParams()
    opts = getOpts()
    # curation.py should be executed once before
    imdb = utils.loadImdbFromPkl(params['curation_path'], params['crops_train'])
    rgbMeanZ, rgbVarZ, rgbMeanX, rgbVarX = loadStats(params['curation_path'])
    imdb, imdbInd = chooseValSet(imdb, opts)

    # random seed should be fixed here
    random.seed(opts['randomSeed'])
    exemplar = tf.placeholder(tf.float32, [params['trainBatchSize'], opts['exemplarSize'], opts['exemplarSize'], 3])
    instance = tf.placeholder(tf.float32, [params['trainBatchSize'], opts['instanceSize'], opts['instanceSize'], 3])

    isTraining = tf.convert_to_tensor(True, dtype='bool', name='is_training')
    score = sn.buildNetwork(exemplar, instance, isTraining)

    restSz = int(score.get_shape()[1])
    restSz = [restSz, restSz]
    respStride = 8  # calculated from stride of convolutional layers and pooling layers
    y = tf.placeholder(tf.float32, [params['trainBatchSize'], restSz[0], restSz[0], 1])




    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(score))

    return




if __name__ == '__main__':
    tf.app.run()

