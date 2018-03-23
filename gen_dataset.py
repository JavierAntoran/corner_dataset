from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt


def gen_figure(imsize, nclass, offset0, offset1, length0, length1, thick0, thick1, noiselvl):

    class_starts = np.array([[0, 0], [imsize - 1, 0], [0, imsize - 1], [imsize - 1, imsize - 1]])
    start = class_starts[nclass]

    if start[0] < imsize/2:
        sign0 = 1
    else:
        sign0 = -1

    if start[1] < imsize/2:
        sign1 = 1
    else:
        sign1 = -1

    length0 *= sign0
    length1 *= sign1

    thick0 = thick0 * sign0
    thick1 = thick1 * sign1

    f = np.zeros((imsize, imsize))

    start[0] += offset0 * sign0
    start[1] += offset1 * sign1

    f[start[0]:start[0]+length0:sign0, start[1]:start[1]+thick1:sign1] = 1
    f[start[0]:start[0]+thick0:sign0, start[1]:start[1]+length1:sign1] = 1

    f += np.sqrt(noiselvl) * np.random.randn(imsize, imsize)

    return f

def shuffle_in_unison(a, b, c=None):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    if c is not None:
        np.random.set_state(rng_state)
        np.random.shuffle(c)


# ---------------
# f = gen_figure(32, 1, 2, 2, 27, 27, 1, 1, 0.05)
# plt.figure()
# plt.imshow(f)
# plt.show()
# -------------------

imgSize = 32
nClasses = 4

sideLength = np.arange(21, 28, 2)
side_offset = np.arange(1, 5, 1)
thick = np.arange(1, 3, 1)
noiseLvl = 0.05


nFigures = nClasses * len(thick)**2 * len(sideLength)**2 * len(side_offset) ** 2
print('generating %d images:' % nFigures)
i = 0

np.random.seed(1337)

figures = np.zeros((nFigures, imgSize, imgSize))
labels = []
y = []
for cl in range(nClasses):
    for s0 in sideLength:
        for s1 in sideLength:
            for so0 in side_offset:
                for so1 in side_offset:
                    for t0 in thick:
                        for t1 in thick:
                            figures[i] = \
                                gen_figure(imgSize, cl, so0, so1, s0, s1, t0, t1, 0.05)
                            labels.append('%d %d %d %d' % (so0, so1, t0, t1))
                            y.append(cl)
                            i += 1
y = np.array(y)
shuffle_in_unison(figures, y, labels)
print('Done!')

np.save('data/angle_ims.npy', figures)
np.save('data/angle_targets.npy', y)

# plots

print('Plotting images')


Ncols = 10
Nrows = 10

fig = plt.figure(figsize=(Ncols, Nrows), dpi=120)
im = []
for i in range(Nrows*Ncols):
    ax1 = fig.add_subplot(Nrows, Ncols, i + 1)
    im.append(ax1.imshow(figures[i, :, :], cmap='gray'))
    ax1.set_title(y[i])
    ax1.axis('off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    plt.subplots_adjust(wspace=0.5, hspace=0.1)
plt.savefig('pics/100_samples.png')
plt.show()
print('Done!')