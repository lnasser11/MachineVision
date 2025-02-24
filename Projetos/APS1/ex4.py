import cv2
import numpy as np
import matplotlib.pyplot as plt

img4_a = cv2.imread('C:\\Users\\lucca\\OneDrive - Insper\\Documentos\\Insper\\6\\VisMaq\\MachineVision\\Projetos\\APS1\\Figuras_APS1\\Fig_APS1_4a.bmp')
img4_b = cv2.imread('C:\\Users\\lucca\\OneDrive - Insper\\Documentos\\Insper\\6\\VisMaq\\MachineVision\\Projetos\\APS1\\Figuras_APS1\\Fig_APS1_4b.bmp')

media_intensidade = np.mean(img4_b)

K = media_intensidade / (img4_b + 1e-6)

img_corrigida = np.clip(img4_a * K, 0, 255).astype(np.uint8)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Imagem Original')
plt.imshow(img4_a, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Padrão de Iluminação')
plt.imshow(img4_b, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Imagem Corrigida')
plt.imshow(img_corrigida, cmap='gray')
plt.axis('off')

plt.show()
