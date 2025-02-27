import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar as imagens
img_1a = cv2.imread('C:\\Users\\lucca\\OneDrive - Insper\\Documentos\\Insper\\6\\VisMaq\\MachineVision\\Projetos\\APS1\\Figuras_APS1\\Fig_APS1_1a.bmp', cv2.IMREAD_GRAYSCALE)
img_1b = cv2.imread('C:\\Users\\lucca\\OneDrive - Insper\\Documentos\\Insper\\6\\VisMaq\\MachineVision\\Projetos\\APS1\\Figuras_APS1\\Fig_APS1_1b.bmp', cv2.IMREAD_GRAYSCALE)

# Subtrair as imagens
img1_diff = cv2.subtract(img_1a, img_1b)

# Somar a diferença com a segunda imagem
img1_add = cv2.add(img1_diff, img_1b)
img1_add = cv2.cvtColor(img1_add, cv2.COLOR_BGR2RGB)

# Converter a imagem somada para escala de cinza
img1_gray = cv2.cvtColor(img1_add, cv2.COLOR_BGR2GRAY)

# Obter dimensões
(h, w) = img1_gray.shape
(a, b) = img_1b.shape

print('Height: ', h)
print('Width = ', w)

# Converter para preto e branco usando um threshold
img1_bnw = np.zeros((h,w), dtype="uint8")
threshold = 105

for i in range(h):
    for j in range(w):
        if img1_gray[i, j] >= threshold:
            img1_bnw[i, j] = 255

# Calcular a proporção de pixels pretos
black = 0
total = 0

for i in range(h):
    for j in range(w):
        if img1_diff[i, j] > 0:
            total += 1

for i in range(h):
    for j in range(w):
        if img1_bnw[i, j] == 0:
            black += 1

percent = (black / total) * 100

print('Pixels pretos:', black)
print('Total de pixels:', total)
print(f'% de nós/falhas = {percent:.2f}%')

# Conversão para RGB para exibição correta com Matplotlib
img_1a_rgb = cv2.cvtColor(img_1a, cv2.COLOR_BGR2RGB)
img_1b_rgb = cv2.cvtColor(img_1b, cv2.COLOR_BGR2RGB)
img1_diff_rgb = cv2.cvtColor(img1_diff, cv2.COLOR_BGR2RGB)

# Plotando as imagens na ordem correta
plt.figure(figsize=(20, 5))

# Imagem Original 1a
plt.subplot(1, 4, 1)
plt.title('Imagem Original 1a')
plt.imshow(img_1a_rgb)
plt.axis('off')

# Imagem Original 1b
plt.subplot(1, 4, 2)
plt.title('Imagem Original 1b')
plt.imshow(img_1b_rgb)
plt.axis('off')

# Imagem Subtraída
plt.subplot(1, 4, 3)
plt.title('Imagem Subtraída')
plt.imshow(img1_diff_rgb)
plt.axis('off')

# Imagem Preto e Branco
plt.subplot(1, 4, 4)
plt.title('Imagem Preto e Branco')
plt.imshow(img1_bnw, cmap='gray')
plt.axis('off')

plt.show()
