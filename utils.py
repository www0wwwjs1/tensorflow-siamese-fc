# Suofei ZHANG, 2017.
# import scipy.io as sio
import numpy as np
import pickle
import h5py
import hdf5storage as hdf

def configParams():
    params = {}

    print("Utils init")

    params['gpuId'] = 0
    params['video'] = ""
    params['numScale'] = 3
    params['scaleStep'] = 1.0375
    params['scalePenalty'] = 0.9745
    params['scaleLR'] = 0.59
    params['responseUp'] = 16
    params['windowing'] = "cosine"
    params['wInfluence'] = 0.176
    params['netFile'] = ""
    params['exemplarSize'] = 127
    params['instanceSize'] = 255
    params['scoreSize'] = 17
    params['totalStride'] = 8
    params['contextAmount'] = 0.5
    params['subMean'] = 0
    params['prefix_z'] = "a_"
    params['prefix_x'] = "b_"
    params['prefix_join'] = "xcorr"
    params['prefix_adj'] = "adjust"
    params['id_feat_z'] = "a_feat"
    params['id_score'] = "score"
    params['net_base_path'] = "models/"
    params['seq_base_path'] = "demo-sequences/"
    params['data_path'] = "/media/zhang/zhang/data/"
    params['curation_path'] = "/media/zhang/work/work/DeepProjs/siamese-net/siamese-fc/ILSVRC15-curation/"

    return params

def loadImdbFromMat(imdbPath):
    # imdb = sio.loadmat(imdbPath)
    # imdbMat = h5py.File(imdbPath)
    imdbMat = hdf.loadmat(imdbPath, 'r')

    imdb = Imdb()

    for i in range(0, imdbMat['id'].shape[0]):
        imdb.id[i] = imdbMat['id'][i][0]
        imdb.n_valid_objects[i] = imdbMat['n_valid_objects'][i][0]
        imdb.nframes[i] = imdbMat['n_valid_objects'][i][0]

        mPath = imdbMat['path'][i][0]

        # mObjs = imdbMat['objects'][i]
        # imdbObjs = None
        # for j in range(0, mObjs.shape[0]):
        #     mObj = mObjs[j]
        #     a = mObj['track_id']
        #     imdbObj = ImdbObject(mObj['track_id'][0], mObj['class'][0], mObjs['valid'][0], mObj['frame_path'][0])












    return imdb

class Imdb:
    id = None
    n_valid_objects = None
    nframes = None
    objects = None
    path = None
    total_valid_objects = 0
    valid_per_trackid = None
    valid_trackids = None

    def __init__(self):
        self.id = np.zeros(4404, dtype=np.uint32)
        self.n_valid_objects = np.zeros(4404, dtype=np.uint32)
        self.nframes = np.zeros(4404, dtype=np.uint32)

class ImdbObject:
    track_id = 0
    mclass = 17
    frames_sz = None
    extent = None
    valid = 0
    frame_path = ""

    def __init__(self, _track_id, _mclass, _valid, _frame_path):
        self.track_id = _track_id
        self.mclass = _mclass
        self.valid = _valid
        self.frame_path = _frame_path
        self.frames_sz = np.zeros(2, dtype=np.uint32)
        self.extent = np.zeros(4, dtype=np.uint32)








def convertImdbFromMat(imdbPath, rawPath):
    imdb = loadImdbFromMat()
    pickle.dump(imdb, open(rawPath, mode='wb'), protocol=pickle.HIGHEST_PROTOCOL)
