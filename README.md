# tensorflow-siamese-fc
A Python+Tensorflow implementation of [siamese-fc](https://github.com/bertinetto/siamese-fc)

- - - -
This is the Python+Tensorflow implementation of Fully-Convolutional Siamese Networks for Object Tracking, including both training and tracking.

The original Matlab version can be found at [siamese-fc](https://github.com/bertinetto/siamese-fc).

The tracker borrows a lot of code from [py-siamese_fc](https://github.com/GreenKing/py-siamese_fc), many thanks.

ref: Fully-Convolutional Siamese Networks for Object Tracking
- - - -

Prerequisites:

python: 3.4; tensorflow: 1.0.1
- - - -

1. [ **Tracking only** ] For tracking, pretrained networks as ckpt file can be plugged into the `tracker.py` directly.
   1. Clone the repository.
   2. A pretrained networks with Imagenet VID dataset can be downloaded from [baidu pan](https://pan.baidu.com/s/1skCcuLZ), unzip the file to `./models/`.
   3. Execute `tracker.py`, the video sequence (`./demo-sequences/vot15_bag`) is processed as an example.
   
2. [ **Training** ] To train a model, following steps can be considered.
   1. Clone the reposistory.
   2. Follow the [instructions from original version](https://github.com/bertinetto/siamese-fc/tree/master/ILSVRC15-curation) to generate the curated dataset for training.
   3. Open the created `imageStats.mat` with MATLAB, upzip `x.mat` and `z.mat` in the path `./ILSVRC15-curation/`, run `curation.py` to get `imdb.pkl` and `imageStats.pkl` as python version of `imdb_video.mat` and `imageStats.mat`.
   4. Execute `train.py` to train your own model. Tensorboard can also be used during this phase to monitor variables in the network.
   5. Parameters of networks are saved as ckpt files in `./ckpt` for tracking.