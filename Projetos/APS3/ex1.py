import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_bordas_brancas(binary_img):
    inverted = cv2.bitwise_not(binary_img.copy())
    h, w = inverted.shape
    mask = np.zeros((h+2, w+2), np.uint8)

    for row in range(h):
        for col in [0, w-1]:
            if inverted[row, col] == 0:
                cv2.floodFill(inverted, mask, (col, row), 255)
    for col in range(w):
        for row in [0, h-1]:
            if inverted[row, col] == 0:
                cv2.floodFill(inverted, mask, (col, row), 255)

    result = cv2.bitwise_not(inverted)
    return result

paths = [
    r"Figuras_APS3\Fig1_caixa1.jpg",
    r"Figuras_APS3\Fig1_caixa2.jpg",
    r"Figuras_APS3\Fig1_caixa3.jpg",
    r"Figuras_APS3\Fig1_caixa4.jpg",
    r"Figuras_APS3\Fig1_caixa5.jpg"
]

x, y, w, h = 80, 120, 610, 740

recortes = {
    "sup_esq": (90, 70, 100, 100),
    "sup_dir": (90, 420, 100, 100),
    "esq_cima": (200, 0, 100, 100),
    "esq_baixo": (550, 10, 100, 100)
}

for path in paths:
    img = cv2.imread(path)
    nome = path.split("\\")[-1].split(".")[0]
    roi = img[y:y + h, x:x + w].copy()

    for label, (dy, dx, hh, ww) in recortes.items():
        subimg = roi[dy:dy + hh, dx:dx + ww]
        gray = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)

        if label in ["sup_esq", "sup_dir"]:
            _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
            processed = cv2.bitwise_not(closed)
            processed = remove_bordas_brancas(processed)
            pix_branco = np.sum(processed == 255)
            tot = processed.size
            prop = pix_branco / tot
            parafuso = prop >= 0.04
            if parafuso:
                print(f'{nome} - {label} - Tem parafuso! (prop = {prop:.4f})')
            else:
                print(f'{nome} - {label} - Não tem parafuso! (prop = {prop:.4f})')

        elif label in ["esq_cima", "esq_baixo"]:
            _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
            processed = closed
            pix_branco = np.sum(processed == 255)
            tot = processed.size
            prop = pix_branco / tot
            parafuso = prop > 0.71
            if parafuso:
                print(f'{nome} - {label} - Tem parafuso! (prop = {prop:.4f})')
            else:
                print(f'{nome} - {label} - Não tem parafuso! (prop = {prop:.4f})')

        # Desenha o retângulo correspondente
        color = (0, 255, 0) if parafuso else (0, 0, 255)  # Verde ou vermelho
        thickness = 2
        top_left = (x + dx, y + dy)
        bottom_right = (x + dx + ww, y + dy + hh)
        cv2.rectangle(img, top_left, bottom_right, color, thickness)

    # Exibe imagem final com os retângulos
    plt.figure(figsize=(8, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'{nome} - Resultado com detecção')
    plt.axis('off')
    plt.show()
