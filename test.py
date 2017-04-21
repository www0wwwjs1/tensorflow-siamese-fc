import utils
import parameters as p
import numpy as np
# import train
import tensorflow.contrib
import tensorflow as tf

# params = p.configParams()

# imdbPath = params['curation_path']+"imdb_video.mat"

# imdb = utils.vidSetupData(params['curation_path'], params['ilsvrc2015'], params['crops_train'])

# imdb = utils.loadImdbFromPkl(params['curation_path'], params['crops_train'])

# for i in range(0, 4404):
#     print (imdb['n_valid_objects'][i][0])
print(tf.__version__)
# test = tf.contrib.image.rotate
print(tf.__path__)
sess = tf.Session()
#
# sess.run(test)
# rgbMeanZ, rgbVarZ, rgbMeanX, rgbVarX = train.loadStats(params['curation_path'])
# imgStats = utils.loadImageStats(params['curation_path'])
# z = imgStats['z']['averageImage'].shape


