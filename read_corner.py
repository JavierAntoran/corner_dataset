import numpy as np
import os
import os.path
import urllib

def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)

def load_train_test_corner():
    dataDir = './data/'
    mkdir(dataDir)
    if not os.path.isfile(dataDir + 'angle_ims.npy'):
        urllib.urlretrieve("https://github.com/JavierAntoran/corner_dataset/blob/master/data/angle_ims.npy",
                            dataDir + 'angle_ims.npy')
        urllib.urlretrieve("https://github.com/JavierAntoran/corner_dataset/blob/master/data/angle_targets.npy",
                       dataDir + 'angle_targets.npy')

    train_ratio = 1 - 0.125

    x = np.load(dataDir + 'angle_ims.npy')
    y = np.load(dataDir + 'angle_targets.npy')
    Ntrain = int(x.shape[0] * train_ratio)

    xtr = np.asarray(x[0:Ntrain], dtype=np.float32)
    xte = np.asarray(x[Ntrain:], dtype=np.float32)
    ytr = np.asarray(y[0:Ntrain], dtype=np.int64)
    yte = np.asarray(y[Ntrain:], dtype=np.int64)

    return xtr, ytr, xte, yte