import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread(r"C:\Users\lucca\OneDrive - Insper\Documentos\Insper\6\VisMaq\MachineVision\Projetos\APS2\Figuras_APS2\Fig1_Tecido1.bmp", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r"C:\Users\lucca\OneDrive - Insper\Documentos\Insper\6\VisMaq\MachineVision\Projetos\APS2\Figuras_APS2\Fig1_Tecido2.bmp", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(r"C:\Users\lucca\OneDrive - Insper\Documentos\Insper\6\VisMaq\MachineVision\Projetos\APS2\Figuras_APS2\Fig1_Tecido3.bmp", cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread(r"C:\Users\lucca\OneDrive - Insper\Documentos\Insper\6\VisMaq\MachineVision\Projetos\APS2\Figuras_APS2\Fig1_Tecido4.bmp", cv2.IMREAD_GRAYSCALE)
img5 = cv2.imread(r"C:\Users\lucca\OneDrive - Insper\Documentos\Insper\6\VisMaq\MachineVision\Projetos\APS2\Figuras_APS2\Fig1_Tecido5.bmp", cv2.IMREAD_GRAYSCALE)

images = [img1, img2, img3, img4, img5]

smoothed_images = []

for img in images:

    kernel_size = 5
    sigma = 1.0
    k = kernel_size // 2
    x = np.linspace(-k, k, kernel_size)
    kernel_1d = np.exp(-0.5 * (x**2) / sigma**2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = np.outer(kernel_1d, kernel_1d)
    

    h, w = img.shape

    pad = kernel_size // 2
    padded_img = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')

    output = np.zeros_like(img)
    

    for i in range(h):
        for j in range(w):
            output[i, j] = np.sum(padded_img[i:i+kernel_size, j:j+kernel_size] * kernel)
    
    smoothed_images.append(output.astype(np.uint8))

images_bin = [cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] for img in smoothed_images]
m = 3
d = int((m-1)/2)

kernel_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]],dtype="int16")
kernel_x = np.array([[1,0,-1],[1,0,-1],[1,0,-1]],dtype="int16")
k = 1
for i, img in enumerate(images_bin):
    (h, w) = img.shape
    fig_out_y = np.zeros((h, w), dtype = 'int16')
    fig_out_x = np.zeros((h, w), dtype = 'int16')
    fig_out_xy = np.zeros((h, w), dtype = 'int16')
    for i in range(d, h-d):
        for j in range(d, w-d):
        
            fig_section_x = img[i-d:i+d+1, j-d:j+d+1] * kernel_x
            sum_x = np.sum(fig_section_x)

            fig_section_y = img[i-d:i+d+1, j-d:j+d+1] * kernel_y
            sum_y = np.sum(fig_section_y)

            fig_out_x[i, j] = sum_x
            fig_out_y[i, j] = sum_y
            fig_out_xy[i, j] = abs(sum_x) + abs(sum_y)

    rows, cols = fig_out_xy.shape
    

    gradient_x = np.abs(fig_out_x)
    gradient_y = np.abs(fig_out_y)

    horizontal_energy = np.sum(gradient_y)
    vertical_energy = np.sum(gradient_x)
    
    total_energy = horizontal_energy + vertical_energy
    horizontal_ratio = horizontal_energy / total_energy if total_energy > 0 else 0
    vertical_ratio = vertical_energy / total_energy if total_energy > 0 else 0
    
    row_score = horizontal_ratio
    col_score = vertical_ratio
    
    diagonal_threshold = 0.4
    
    if horizontal_ratio > 0.6 and vertical_ratio < 0.4:
        pattern = "padrão horizontal"
    elif vertical_ratio > 0.6 and horizontal_ratio < 0.4:
        pattern = "padrão vertical"
    elif abs(horizontal_ratio - vertical_ratio) < 0.2:
        pattern = "padrão diagonal"
    else:
        pattern = "padrão horizontal" if horizontal_ratio > vertical_ratio else "padrão vertical"
    
    print(f"Imagem {k}: {pattern} (Row score: {row_score:.2f}, Col score: {col_score:.2f})")
    k += 1
    plt.subplot(1, 4, 1)
    plt.imshow(fig_out_x, cmap='gray')
    plt.title('X')
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(fig_out_y, cmap='gray')
    plt.title('Y')
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(fig_out_xy, cmap='gray')
    plt.title('XY')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(img, cmap='gray')
    plt.title('Imagem original filtrada')
    plt.axis('off')
    plt.show()
