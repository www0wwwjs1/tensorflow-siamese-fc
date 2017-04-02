# Copyright (c) <2017> <SUOFEI ZHANG>. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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