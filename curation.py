# Suofei ZHANG, 2017.

import parameters as p
import utils

def main():

    print('ilsvrc2015 curation...')

    params = p.configParams()

    imdbPath = params['curation_path']+"imdb_video.mat"
    imdb = utils.vidSetupData(params['curation_path'], params['ilsvrc2015'], params['crops_train'])
    imdb = utils.loadImdbFromPkl(params['curation_path'], params['crops_train'])

    imageStats = utils.loadImageStatsFromMat(params['curation_path'])

if __name__=='__main__':
        main()