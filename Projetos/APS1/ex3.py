

import cv2
import numpy as np
import matplotlib.pyplot as plt

fig_2 = cv2.imread('Figuras/Fig0342a.tif', cv2.IMREAD_GRAYSCALE)

m = 3
d = int((m-1)/2)

kernel_x = np.array([ [-1, -1, -1], [0, 0, 0], [1, 1, 1] ], dtype = 'int16')
kernel_y = np.array([ [-1, 0, 1], [-1, 0, 1], [-1, 0, 1] ], dtype = 'int16')

(h, w) = fig_2.shape

fig_out = np.zeros((h, w), dtype = 'uint8')

for i in range(d, h-d):
    for j in range(d, w-d):

        fig_section_x = fig_2[i-d:i+d+1, j-d:j+d+1] * kernel_x
        fig_section_y = fig_2[i-d:i+d+1, j-d:j+d+1] * kernel_y

        fig_out[i, j] = abs(np.sum(fig_section_x)) + abs(np.sum(fig_section_y))

plt.figure(figsize=(10,10))

plt.subplot(1,2,1)
plt.imshow(fig_2, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(fig_out, cmap='gray')
plt.title('Filtrada Final')
plt.axis('off')

plt.show()
