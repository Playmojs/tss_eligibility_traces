import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import utils

shear_matrix = np.array([[1, 0.44], [0, 1]])
inverse_shear = np.linalg.inv(shear_matrix)
size = 4
ly = 1
pxs = 48
dists = np.dot(inverse_shear, np.array([ly, ly]))
sizes = np.dot(inverse_shear, np.array([size, size]))

dots = np.meshgrid(np.arange(0, sizes[0], dists[0]), np.arange(0, sizes[1], dists[1]))
dots = np.reshape(dots, (2,-1))

hex_field = utils.createHexField(pxs, 0.35, 0, 'square', True).T
plt.imshow(hex_field, origin = 'lower')
plt.show()

data_max = ndimage.maximum_filter(hex_field, size = 2*pxs*0.108, axes=(-2,-1))
maxima = np.nonzero(hex_field==data_max)


maxima = np.array([maxima[1][maxima[0] < 45], maxima[0][maxima[0]<45]])

fig, ax = plt.subplots()
ax.scatter(maxima[0], maxima[1])
ax.set_aspect('equal')
ax.set_xlim([0, pxs])
ax.set_ylim([0, pxs])
plt.show()


re_dots = inverse_shear @ maxima

fig, ax = plt.subplots()
ax.scatter(re_dots[0], re_dots[1])
ax.set_aspect('equal')
plt.show()

reduced_dots_x = re_dots[0] % 14
reduced_dots_y = re_dots[1] % 16

fig, ax = plt.subplots()
ax.scatter(reduced_dots_x, reduced_dots_y)
ax.set_aspect('equal')
ax.set_xlim([0, 14])
ax.set_ylim([0, 16])
plt.show()

# fig, ax = plt.subplots()
# ax.scatter(dots[0], dots[1])
# ax.set_aspect('equal')
# plt.show()

# sheared_dots = inverse_shear @ dots

# fig, ax = plt.subplots()
# ax.scatter(sheared_dots[0], sheared_dots[1])
# ax.set_aspect('equal')
# plt.show()

