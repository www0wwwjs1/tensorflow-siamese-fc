# Suofei ZHANG, 2017.

class Params:
    opts = {
        'video': "",
        'numScale': 3,  #only 3 scales are considered here
        'scaleStep': 1.0375,
        'scalePenalty': 0.9745,
        'scaleLR': 0.59,
        'responseUp': 16,
        'windowing': 'cosine',
        'wInfluence': 0.176,
        'netFile': "",
        'exemplarSize': 127,
        'instanceSize': 255,
        'scoreSize': 17,
        'totalStride': 8,
        'contextAmount': 0.5,
        'subMean': 0,
        'prefix_z': "a_",
        'prefix_x': "b_",
        'prefix_join': "xcorr",
        'prefix_adj': "adjust",
        'id_feat_z': "a_feat",
        'id_score': "score",
        'net_base_path': "models/",
        'seq_base_path': "demo-sequences/"
    }

    def __init__(self, argvs=[]):
        print("Utils init")