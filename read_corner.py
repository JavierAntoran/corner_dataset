import numpy as np
import os
import os.path

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
        os.system(
            'wget https://github.com/JavierAntoran/corner_dataset/blob/master/data/angle_ims.npy -O ./data/angle_ims.npy')
        os.system(
            'wget https://github.com/JavierAntoran/corner_dataset/blob/master/data/angle_ims.npy -O ./data/angle_targets.npy')

    train_ratio = 1 - 0.125

    x = np.load(dataDir + 'angle_ims.npy')
    y = np.load(dataDir + 'angle_targets.npy')
    Ntrain = int(x.shape[0] * train_ratio)

    xtr = x[0:Ntrain]
    xte = x[Ntrain:]
    ytr = y[0:Ntrain]
    yte = y[Ntrain:]

    return xtr, ytr, xte, yte
