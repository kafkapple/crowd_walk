# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:31:52 2018

@author: 2014_Joon_IBS
"""

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2, 2)

axs[0, 0].imshow(np.random.random((100, 100)))

axs[0, 1].imshow(np.random.random((100, 100)))

axs[1, 0].imshow(np.random.random((100, 100)))

axs[1, 1].imshow(np.random.random((100, 100)))

plt.subplot_tool()
plt.show()