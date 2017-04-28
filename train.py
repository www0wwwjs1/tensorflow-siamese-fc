# Suofei ZHANG, 2017.

import numpy as np
from numpy.matlib import repmat
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

def createLogLossLabel(labelSize, rPos, rNeg):
    labelSide = labelSize[0]

    logLossLabel = np.zeros(labelSize, dtype=np.float32,)
    labelOrigin = np.array([np.floor(labelSide/2), np.floor(labelSide/2)])

    for i in range(0, labelSide):
        for j in range(0, labelSide):
            distFromOrigin = np.linalg.norm(np.array([i, j])-labelOrigin)
            if distFromOrigin <= rPos:
                logLossLabel[i, j] = 1
            else:
                if distFromOrigin <= rNeg:
                    logLossLabel[i, j] = 0
                else:
                    logLossLabel[i, j] = -1

    return logLossLabel

def createLabels(labelSize, rPos, rNeg):
    half = np.floor(labelSize[0]/2)

    fixedLabel = createLogLossLabel(labelSize, rPos, rNeg)
    instanceWeight = np.ones(fixedLabel.shape)
    idxP = np.where(fixedLabel == 1)
    idxN = np.where(fixedLabel == -1)

    sumP = len(idxP[0])
    sumN = len(idxN[0])

    instanceWeight[idxP[0], idxP[1]] = 0.5*instanceWeight[idxP[0], idxP[1]]/sumP
    instanceWeight[idxN[0], idxN[1]] = 0.5*instanceWeight[idxN[0], idxN[1]]/sumN

    return fixedLabel, instanceWeight

def precisionAuc(positions, groundTruth, radius, nStep):
    thres = np.linspace(0, radius, nStep)

    errs = np.zeros([nStep], dtype=np.float32)

    distances = np.sqrt(np.power(positions[:, 0]-groundTruth[:, 0], 2)+np.power(positions[:, 1]-groundTruth[:, 1], 2))
    distances[np.where(np.isnan(distances))] = []

    for p in range(0, nStep):
        errs[p] = np.shape(np.where(distances > thres[p]))[-1]

    score = np.trapz(errs)

    return score

def centerThrErr(score, labels, oldRes, m):
    radiusInpix = 50
    totalStride = 8
    nStep = 100
    batchSize = np.shape(score)[0]
    posMask = np.where(labels > 0)
    numPos = np.shape(posMask)[-1]

    res = 0

    responses = np.squeeze(score[posMask, :, :, :], axis=(0,))
    half = np.floor(np.shape(score)[1]/2)
    centerLabel = repmat([half, half], numPos, 1)
    positions = np.zeros([numPos, 2], dtype=np.float32)

    for b in range(0, numPos):
        sc = np.squeeze(responses[b, :, :, 0])
        r = np.where(sc == np.max(sc))
        positions[b, :] = [r[0][0], r[1][0]]

    res = precisionAuc(positions, centerLabel, radiusInpix/totalStride, nStep)

    res = (oldRes*m+res)/(m+batchSize)
    m = m+batchSize
    return res, m

def centerScore(x):
    m1, m2 = np.shape(x)
    c1 = (m1+1)/2-1
    c2 = (m2+1)/2-1
    v = x[int(c1), int(c2)]

    return v

def maxScoreErr(x, yGt, oldRes, m):
    b, m1, m2, k = np.shape(x)

    errs = np.zeros([b], dtype=np.float32)

    for i in range(0, b):
        score = np.squeeze(x[i, :, :, 0])

        if yGt[i] > 0:
            errs[i] = centerScore(score)
        else:
            errs[i] = -np.max(score)

    res = len(np.where(errs <= 0)[0])

    res = (oldRes*m+res)/(m+b)
    m = m+b

    return res, m

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
    # labels = tf.placeholder(tf.float32, [params['trainBatchSize']])

    isTraining = tf.convert_to_tensor(True, dtype='bool', name='is_training')
    score = sn.buildNetwork(exemplar, instance, isTraining)

    respSz = int(score.get_shape()[1])
    respSz = [respSz, respSz]
    respStride = 8  # calculated from stride of convolutional layers and pooling layers
    fixedLabel, instanceWeight = createLabels(respSz, opts['lossRPos']/respStride, opts['lossRNeg']/respStride)
    instanceWeight = tf.constant(instanceWeight, dtype=tf.float32)

    y = tf.placeholder(tf.float32, [params['trainBatchSize'], respSz[0], respSz[0], 1])

    loss = tf.reduce_mean(sn.loss(score, y, instanceWeight))





    score = np.zeros([8, 15, 15, 1], dtype=np.float32)
    labels = np.ones([8], dtype=np.float32)
    for b in range(0, 8):
        for i in range(0, 15):
            for j in range(0, 15):
                score[b, i, j, 0] = np.random.randn()

    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # print(sess.run(score))
    errDispNum = 0
    errDisp = 0
    errMaxNum = 0
    errMax = 0

    errDisp = centerThrErr(score, labels, errDisp, errDispNum)
    errMax = maxScoreErr(score, labels, errMax, errMaxNum)


    return




if __name__ == '__main__':
    tf.app.run()

