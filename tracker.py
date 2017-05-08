# Suofei ZHANG, 2017.

import os
import tensorflow as tf
import numpy as np
import glob
import matplotlib.image as mpimg
# from PIL import Image
from skimage import transform

import utils
from siamese_net import SiameseNet
from parameters import configParams

def getOpts(opts):
    print("config opts...")

    opts['numScale'] = 3
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
    opts['subMean'] = False

    opts['video'] = 'vot15_bag'
    opts['modelPath'] = './models/'
    opts['modelName'] = opts['modelPath']+"model_epoch30.ckpt"
    opts['summaryFile'] = './data_track/'+opts['video']+'_3'

    return opts

def getAxisAlignedBB(region):
    region = np.array(region)
    nv = region.size
    assert (nv == 8 or nv == 4)

    if nv == 8:
        xs = region[0 : : 2]
        ys = region[1 : : 2]
        cx = np.mean(xs)
        cy = np.mean(ys)
        x1 = min(xs)
        x2 = max(xs)
        y1 = min(ys)
        y2 = max(ys)
        A1 = np.linalg.norm(np.array(region[0:2])-np.array(region[2:4]))*np.linalg.norm(np.array(region[2:4])-np.array(region[4:6]))
        A2 = (x2-x1)*(y2-y1)
        s = np.sqrt(A1/A2)
        w = s*(x2-x1)+1
        h = s*(y2-y1)+1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2

    return cx-1, cy-1, w, h

def frameGenerator(vpath):
    imgs = []
    for imgFile in glob.glob(os.path.join(vpath, "*.jpg")):
        imgs.append(mpimg.imread(imgFile).astype(np.float))
        # imgs.append(np.array(Image.open(imgFile)).astype(np.float32))

    return imgs

def loadVideoInfo(basePath, video):
    videoPath = os.path.join(basePath, video, 'imgs')
    groundTruthFile = os.path.join(basePath, video, 'groundtruth.txt')

    groundTruth = open(groundTruthFile, 'r')
    reader = groundTruth.readline()
    region = [float(i) for i in reader.strip().split(",")]
    cx, cy, w, h = getAxisAlignedBB(region)
    pos = [cy, cx]
    targetSz = [h, w]

    imgs = frameGenerator(videoPath)

    return imgs, np.array(pos), np.array(targetSz)

def getSubWinTracking(img, pos, modelSz, originalSz, avgChans):
    if originalSz is None:
        originalSz = modelSz

    sz = originalSz
    im_sz = img.shape
    # make sure the size is not too small
    assert min(im_sz[:2]) > 2, "the size is too small"
    c = (np.array(sz) + 1) / 2

    # check out-of-bounds coordinates, and set them to black
    context_xmin = round(pos[1] - c[1])
    context_xmax = context_xmin + sz[1] - 1
    context_ymin = round(pos[0] - c[0])
    context_ymax = context_ymin + sz[0] - 1
    left_pad = max(0, int(-context_xmin))
    top_pad = max(0, int(-context_ymin))
    right_pad = max(0, int(context_xmax - im_sz[1] + 1))
    bottom_pad = max(0, int(context_ymax - im_sz[0] + 1))

    context_xmin = int(context_xmin + left_pad)
    context_xmax = int(context_xmax + left_pad)
    context_ymin = int(context_ymin + top_pad)
    context_ymax = int(context_ymax + top_pad)

    if top_pad or left_pad or bottom_pad or right_pad:
        r = np.pad(img[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=avgChans[0])
        g = np.pad(img[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=avgChans[1])
        b = np.pad(img[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=avgChans[2])
        img = np.concatenate((r, g, b), axis=2)

    im_patch_original = img[context_ymin:context_ymax + 1, context_xmin:context_xmax + 1, :]
    if not np.array_equal(modelSz, originalSz):
        im_patch_original = im_patch_original/255.0
        im_patch = transform.resize(im_patch_original, modelSz)*255.0
        # im = Image.fromarray(im_patch_original.astype(np.float))
        # im = im.resize(modelSz)
        # im_patch = np.array(im).astype(np.float32)
    else:
        im_patch = im_patch_original

    return im_patch, im_patch_original

'''----------------------------------------main-----------------------------------------------------'''
def main(_):
    print('run tracker...')
    opts = configParams()
    opts = getOpts(opts)

    exemplarOp = tf.placeholder(tf.float32, [1, opts['exemplarSize'], opts['exemplarSize'], 3])
    instanceOp = tf.placeholder(tf.float32, [opts['numScale'], opts['instanceSize'], opts['instanceSize'], 3])

    sn = SiameseNet()
    zFeatOp = sn.buildExemplarSubNetwork(exemplarOp, opts)

    writer = tf.summary.FileWriter(opts['summaryFile'])
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, opts['modelName'])

    zFeatConstantOp = tf.get_variable('zFeatConstant', [6, 6, 256, 1], dtype=tf.float32)
    score = sn.buildInferenceNetwork(instanceOp, zFeatConstantOp, opts)

    writer.add_graph(sess.graph)

    weights = sess.graph.get_operation_by_name("adjust/weights")
    test = weights.values()
    weights = sess.run(test)

    imgs, targetPosition, targetSize = loadVideoInfo(opts['seq_base_path'], opts['video'])
    nImgs = len(imgs)
    startFrame = 0

    img = imgs[startFrame]
    if(img.shape[-1] == 1):
        tmp = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.float)
        tmp[:, :, 0] = tmp[:, :, 1] = tmp[:, :, 2] = np.squeeze(img)
        img = tmp

    avgChans = np.mean(img, axis=(0, 1))# [np.mean(np.mean(img[:, :, 0])), np.mean(np.mean(img[:, :, 1])), np.mean(np.mean(img[:, :, 2]))]
    wcz = targetSize[1]+opts['contextAmount']*np.sum(targetSize)
    hcz = targetSize[0]+opts['contextAmount']*np.sum(targetSize)
    sz = np.sqrt(wcz*hcz)
    scalez = opts['exemplarSize']/sz

    zCrop, _ = getSubWinTracking(img, targetPosition, [opts['exemplarSize'], opts['exemplarSize']], [np.around(sz), np.around(sz)], avgChans)

    if opts['subMean']:
        pass

    dSearch = (opts['instanceSize']-opts['exemplarSize'])/2
    pad = dSearch/scalez
    sx = sz+2*pad

    minSz = 0.2*sx
    maxSx = 5.0*sx

    winSz = opts['scoreSize']*opts['responseUp']
    if opts['windowing'] == 'cosine':
        hann = np.hanning(winSz).reshape(winSz, 1)
        window = hann.dot(hann.T)
    elif opts['windowing'] == 'uniform':
        window = np.ones((winSz, winSz), dtype=np.float32)

    window = window/np.sum(window)
    scales = np.array([opts['scaleStep'] ** i for i in range(int(np.ceil(opts['numScale']/2.0)-opts['numScale']), int(np.floor(opts['numScale']/2.0)+1))])

    zCrop = np.expand_dims(zCrop, axis=0)
    zFeat = sess.run(zFeatOp, feed_dict={exemplarOp: zCrop})
    zFeat = np.transpose(zFeat, [1, 2, 3, 0])
    sess.run(zFeatConstantOp.assign(zFeat))

    tf.Graph().as_default()
    sn = SiameseNet()

    print(zFeat)





if __name__=='__main__':
    tf.app.run()

    # saver = tf.train.import_meta_graph(opts['modelName']+'.meta')
    # graph = tf.get_default_graph()
    # weights = graph.get_tensor_by_name("weights:read:0")


    # zFeat = sess.run(zFeatOp, feed_dict={exemplarOp: exemplar})
    # print(zFeat)
