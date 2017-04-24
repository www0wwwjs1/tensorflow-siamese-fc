# Suofei ZHANG, 2017.

import numpy as np
import tensorflow as tf
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

def main(_):
    params = configParams()
    opts = getOpts()
    # curation.py should be executed once before
    imdb = utils.loadImdbFromPkl(params['curation_path'], params['crops_train'])

    rgbMeanZ, rgbVarZ, rgbMeanX, rgbVarX = loadStats(params['curation_path'])

    # random seed should be fixed here
    exemplar = tf.placeholder(tf.float32, [1, opts['exemplarSize'], opts['exemplarSize'], 3])
    instance = tf.placeholder(tf.float32, [3, opts['exemplarSize'], opts['exemplarSize'], 3])

    isTraining = tf.convert_to_tensor(True, dtype='bool', name='is_training')
    score = sn.buildNetwork(exemplar, instance, isTraining)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    print(sess.run(score))

    return


if __name__ == '__main__':
    tf.app.run()

