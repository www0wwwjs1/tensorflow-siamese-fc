import utils
import parameters as p
import numpy as np
import tensorflow as tf

params = p.configParams()

# imdbPath = params['curation_path']+"imdb_video.mat"

# imdb = utils.vidSetupData(params['curation_path'], params['ilsvrc2015'], params['crops_train'])

# imdb = utils.loadImdbFromPkl(params['curation_path'], params['crops_train'])

# for i in range(0, 4404):
#     print (imdb['n_valid_objects'][i][0])

test = tf.placeholder('bool', [], name='is_training')

sess = tf.Session()

sess.run(test)

print 'aaa'

