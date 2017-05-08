def configParams():
    params = {}

    print("config parameters...")

    params['gpuId'] = 0
    params['data_path'] = "/media/zhang/zhang/data/"
    params['ilsvrc2015'] = params['data_path']+"ILSVRC2015_VID/"
    params['crops_path'] = params['data_path']+"ILSVRC2015_CROPS/"
    params['crops_train'] = params['crops_path']+"Data/VID/train/"
    params['curation_path'] = "./ILSVRC15-curation/"
    params['seq_base_path'] = "./demo-sequences/"
    params['trainBatchSize'] = 8
    params['numScale'] = 3




    return params


    # params['video'] = ""
    # params['scaleStep'] = 1.0375
    # params['scalePenalty'] = 0.9745
    # params['scaleLR'] = 0.59
    # params['responseUp'] = 16
    # params['windowing'] = "cosine"
    # params['wInfluence'] = 0.176
    # params['netFile'] = ""
    # params['exemplarSize'] = 127
    # params['instanceSize'] = 255
    # params['scoreSize'] = 17
    # params['totalStride'] = 8
    # params['contextAmount'] = 0.5
    # params['subMean'] = 0
    # params['prefix_z'] = "a_"
    # params['prefix_x'] = "b_"
    # params['prefix_join'] = "xcorr"
    # params['prefix_adj'] = "adjust"
    # params['id_feat_z'] = "a_feat"
    # params['id_score'] = "score"
    # params['net_base_path'] = "models/"
    # params['seq_base_path'] = "demo-sequences/"
