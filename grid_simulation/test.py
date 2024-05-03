import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import utils
from matplotlib.patches import Arc
pxs = 48
sigma = 0.15

hex_field = utils.createHexField(pxs, 3*sigma, 16, 'square', False).T
autocorr = utils.autoCorr(hex_field)
gscore = utils.gridnessScore(autocorr, pxs, sigma)

dim0 = autocorr.shape[-2]
cntr = dim0 // 2
ring_start = cntr//2 - 5

# Create the ring filter
in_ra = int(2 * sigma * pxs)
out_ra = int(4 * sigma * pxs)
RingFilt = np.full((dim0, dim0), np.nan)
for i in range(dim0):
    for j in range(dim0):
        cntr_i = (cntr - i) ** 2
        cntr_j = (cntr - j) ** 2
        dist = cntr_i + cntr_j
        if in_ra ** 2 <= dist <= out_ra ** 2:
            RingFilt[i, j] = 1

Tmp = autocorr * RingFilt[np.newaxis, ...]

origo = np.array([8, 7])
fig1, ax1 = plt.subplots()
ax1.imshow(hex_field, origin = 'lower')
ax1.axis('off')
ax1.plot([origo[0], 26], [origo[1], 13], color = 'Black', marker = 'h', linewidth = 2.5, markersize = 3)
ax1.plot([origo[0], 15.5], [origo[1], origo[1]], color = 'Black', linewidth = 2)
ax1.plot([0, origo[0]], [0, origo[1]], color = 'Black', linestyle = 'dashed', linewidth = 2)
ax1.add_patch(Arc([origo[0], origo[1]], 15, 15, angle = 0, theta1 = 0, theta2 = 16, linewidth = 2))

ax1.text(2.5, 4.5, '1)', size = 13)
ax1.text(15, 11, '2)', size = 13)
ax1.text(16, 6, '3)', size = 13)

fig2, ax2 = plt.subplots()
ax2.imshow(autocorr, origin = 'lower')
ax2.plot([cntr, 18 + cntr], [cntr, 6 + cntr], color = 'Black', marker = 'h', linewidth = 2.5, markersize = 3)
ax2.plot([cntr, 7.5 + cntr], [cntr, cntr], color = 'Black', linewidth = 2)
ax2.add_patch(Arc([cntr, cntr], 15, 15, angle = 0, theta1 = 0, theta2 = 16, linewidth = 2))


fig3, ax3 = plt.subplots()
ax3.imshow(Tmp[0, ring_start:dim0 - ring_start, ring_start:dim0 - ring_start])
ax3.axis('off')

hex_field2 = utils.createHexField(pxs, 3*sigma, 41, 'square', False).T

inds = np.arange(48 * 48, dtype = int)
fig4, ax4 = plt.subplots(nrows = 2, ncols = 2)
ax4[0, 0].imshow(hex_field, origin = 'lower') 
ax4[0, 1].imshow(hex_field2, origin = 'lower')
for i, hex in enumerate([hex_field, hex_field2]):
    np.random.shuffle(inds)
    shuf_hex = np.ndarray.flatten(hex)[inds]
    ax4[1, i].imshow(np.reshape(shuf_hex, (48, 48)), origin = 'lower')
    ax4[1, i].axis('off')
    
    ax4[0, i].set_xticks([])
    ax4[0, i].set_yticks([])
    ax4[0, i].set_title(f"t{i + 1}")

plt.show()

