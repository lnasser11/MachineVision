import cv2
import numpy as np
import matplotlib.pyplot as plt

img2_a = cv2.imread('C:\\Users\\lucca\\OneDrive - Insper\\Documentos\\Insper\\6\\VisMaq\\MachineVision\\Projetos\\APS1\\Figuras_APS1\\Fig_APS1_2a.bmp', cv2.IMREAD_COLOR)
img2_b = cv2.imread('C:\\Users\\lucca\\OneDrive - Insper\\Documentos\\Insper\\6\\VisMaq\\MachineVision\\Projetos\\APS1\\Figuras_APS1\\Fig_APS1_2b.bmp', cv2.IMREAD_GRAYSCALE)

_, mask = cv2.threshold(img2_b, 250, 255, cv2.THRESH_BINARY_INV)

img2_a_rbg = cv2.cvtColor(img2_a, cv2.COLOR_BGR2RGB)
img2_b_rbg = cv2.cvtColor(img2_b, cv2.COLOR_BGR2RGB)

ovni_resize = cv2.resize(img2_b, (img2_a.shape[1], img2_a.shape[0]))
mask = cv2.resize(mask, (img2_a.shape[1], img2_a.shape[0]))

mask_inv = cv2.bitwise_not(mask)

egito_bg = cv2.bitwise_and(img2_a_rbg, img2_a_rbg, mask=mask_inv)
egito_fg = cv2.bitwise_and(img2_b_rbg, img2_b_rbg, mask=mask)

img_final = cv2.add(egito_bg, egito_fg)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title('Imagem Original')
plt.imshow(img2_a_rbg)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Imagem OVNI')
plt.imshow(img2_b_rbg, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Imagem Somada')
plt.imshow(img_final)
plt.axis('off')

plt.show()
