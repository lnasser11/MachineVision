import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_image(path):
    imagem_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Carrega em escala de cinza
    return imagem_gray

def bin(img):
    img_bin = img/255
    return img_bin

def find_best_matches(image, pattern, top_n=3):
    img_array = np.array(image)
    pat_array = np.array(pattern)
    h_img, w_img = img_array.shape
    h_pat, w_pat = pat_array.shape
    
    match_scores = []
    
    for y in range(h_img - h_pat + 1):
        for x in range(w_img - w_pat + 1):
            region = img_array[y:y+h_pat, x:x+w_pat]
            numerador = np.sum(region * pat_array)
            denominador = (np.sum((region**2))*np.sum((pat_array**2)))**(1/2)
            diff = numerador/denominador
            match_scores.append((diff, x, y))
    
    best_matches = sorted(match_scores,reverse=True)[:top_n]  # Seleciona os top_n melhores
    return [(x, y) for _, x, y in best_matches]

def highlight_matches(image, matches, pattern_size):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    w_pat, h_pat = pattern_size
    for x, y in matches:
        rect = plt.Rectangle((x, y), w_pat, h_pat, edgecolor='red', linewidth=2, fill=False)
        ax.add_patch(rect)
    plt.show()
    
image_path = r"C:\Users\lucca\OneDrive - Insper\Documentos\Insper\6\VisMaq\MachineVision\Projetos\APS2\Figuras_APS2\Fig2_Ferramentas_u8.bmp"
pattern_path = r"C:\Users\lucca\OneDrive - Insper\Documentos\Insper\6\VisMaq\MachineVision\Projetos\APS2\Figuras_APS2\Fig2_Padrao_u8.bmp"

imagem = load_image(image_path)
patternA = load_image(pattern_path)
image = bin(imagem)
pattern = bin(patternA)

matches = find_best_matches(image, pattern)
print(matches)
highlight_matches(image, matches, pattern.shape)

