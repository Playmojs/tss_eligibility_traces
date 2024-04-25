import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import utils

pxs = 48
sigma = 0.1

hex_field = utils.createHexField(pxs, 3*sigma, 7.5, 'square', True).T
autocorr = utils.autoCorr(hex_field)
gscore = utils.gridnessScore(autocorr, pxs, sigma)

fig, ax = plt.subplots(ncols = 2)

ax[0].imshow(hex_field, origin = 'lower')

ax[1].imshow(autocorr, origin = 'lower')
ax[1].set_title(f'{gscore[0]:.2f}')

plt.show()

