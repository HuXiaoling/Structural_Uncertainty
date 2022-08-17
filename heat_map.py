from matplotlib import image as mpimg
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import cv2
from PIL import Image

mask = np.zeros((194, 188, 9))
variance = np.zeros((194, 188))

for i in range(9):
    mask[:,:,i] = mpimg.imread('heat_map/img10_1_2_sample' + str(i+1) + '_final.png')

for j in range(194):
    for k in range(188):
        variance[j, k] = np.var(mask[j,k,:])

variance = variance/np.max(variance)

# variance = mpimg.imread('heat_map/img9_1_2_pred.png')
# for j in range(194):
#     for k in range(188):
#         if (variance[j,k] >= 0.5):
#             variance[j, k] = 1 - variance[j,k]

variance = variance/np.max(variance)

np.save('heatmap.npy', variance)

ax = sns.heatmap(variance, cmap=plt.cm.coolwarm,
         vmin=0, vmax=1)
ax.set_axis_off()

plt.show()
plt.savefig('./heat_map/img10_1_2_heatmap.png', bbox_inches='tight', pad_inches=0)

# img = cv2.imread('heat_map/img9_1_2_ori.png') 

# heatmap = (variance *255).astype(np.uint8)

# heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# # import pdb; pdb.set_trace()
# super_imposed_img = cv2.addWeighted(heatmap_img, 0.4, img, 0.6, 0)
# super_imposed_img = Image.fromarray(super_imposed_img)
# super_imposed_img.save('./heat_map/img9_1_2_overlay.png')



