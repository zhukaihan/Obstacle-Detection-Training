import numpy as np

import scipy.misc
import matplotlib.pylab as plt

imgs = ["20190414122857", "20190414122901", "20190414122908", "20190414122910", "20190414122923"]
for img in imgs:
    image1 = plt.imread(img + ".png")
    scipy.misc.imsave(img + '_orig.jpg', image1[:, :, :3])
    scipy.misc.imsave(img + '_depth.jpg', image1[:, :, 3])