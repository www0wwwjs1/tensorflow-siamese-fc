import utils

params = utils.configParams()

imdbPath = params['curation_path']+"imdb_video.mat"

imdb = utils.loadImdbFromMat(imdbPath)

for i in range(0, 4404):
    print (imdb['n_valid_objects'][i][0])

print ("aaa")

