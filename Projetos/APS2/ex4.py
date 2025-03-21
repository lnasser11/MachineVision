import cv2
import numpy as np
import matplotlib.pyplot as plt

persp1 = cv2.imread('Figuras_APS2/Fig4_Campo_Persp1.bmp', cv2.IMREAD_GRAYSCALE)
persp2 = cv2.imread('Figuras_APS2/Fig4_Campo_Persp2.bmp', cv2.IMREAD_GRAYSCALE)
persp3 = cv2.imread('Figuras_APS2/Fig4_Campo_Persp3.bmp', cv2.IMREAD_GRAYSCALE)
persp4 = cv2.imread('Figuras_APS2/Fig4_Campo_Persp4.bmp', cv2.IMREAD_GRAYSCALE)


import numpy as np
import cv2

def detect_white_dots_centroid(img):
    h, w = img.shape
    visited = np.zeros_like(img, dtype=bool)
    positions = []

    for v in range(h):
        for u in range(w):
            if img[v, u] == 255 and not visited[v, u]:
                # Iniciar um novo grupo (bolinha)
                stack = [(u, v)]
                blob_pixels = []

                while stack:
                    x, y = stack.pop()
                    if (0 <= x < w) and (0 <= y < h) and not visited[y, x] and img[y, x] == 255:
                        visited[y, x] = True
                        blob_pixels.append((x, y))

                        # Checar os 4 vizinhos
                        stack.extend([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])

                # Pegar o centro do grupo
                if blob_pixels:
                    avg_x = int(np.mean([p[0] for p in blob_pixels]))
                    avg_y = int(np.mean([p[1] for p in blob_pixels]))
                    positions.append((avg_x, avg_y))

    return positions

def img_perspective(img, matrix):
    (h, w) = img.shape

    TH_01 = np.array(matrix, dtype=np.float32)
    TH_10 = np.linalg.inv(TH_01)

    img_out = np.ones((h, w), dtype=np.uint8) * 255

    for u in range(w):
        for v in range(h):
            p1 = np.array([[u], [v], [1]])
            p0 = np.matmul(TH_10, p1)

            x = int(p0[0][0] / p0[2][0])
            y = int(p0[1][0] / p0[2][0])

            if 0 <= x < w and 0 <= y < h:
                img_out[v, u] = img[y, x]

    img_out_color = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)

    # Encontrar a posição da linha vermelha
    red_line_pos = None
    for u in range(w):
        if np.any(img_out[:, u] < 50):
            red_line_pos = u
            cv2.line(img_out_color, (u, 0), (u, h - 1), (0, 0, 255), 1)
            break

    # Cortar imagem
    img_out_cropped = img_out[10:490, 10:400]
    img_out_color_cropped = img_out_color[10:490, 10:400]

    # Detectar jogadores (agora pegando o centro de cada bolinha)
    player_positions = detect_white_dots_centroid(img_out_cropped)

    # Contar jogadores à esquerda da linha
    jogadores_a_esquerda = 0
    for u, v in player_positions:
        if red_line_pos is not None and (u + 10) < red_line_pos:
            jogadores_a_esquerda += 1

        # Desenhar bolinha azul na imagem, agora **só no centro da bolinha**
        cv2.circle(img_out_color_cropped, (u, v), 6, (255, 0, 0), -1)

    # Resultado
    if jogadores_a_esquerda >= 2:
        print("Não impedido")
    else:
        print("Impedido")

    print("\nPosições dos jogadores:", player_positions)
    print("\nJogadores à esquerda da linha:", jogadores_a_esquerda)

    # Mostrar imagem
    cv2.imshow('Transformed Image with Players', img_out_color_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





matriz_do_mal = np.array([[101, 11, 1, 0, 0, 0, -101*0, -11*0],
                          [600, 21, 1, 0, 0, 0, -600*400, -21*400],
                          [690, 478, 1, 0, 0, 0, -690*400, -478*400],
                          [5, 218, 1, 0, 0, 0, -5*0, -218*0],
                          [0, 0, 0, 101, 11, 1, -101*0, -11*0],
                          [0, 0, 0, 600, 21, 1, -600*0, -21*0],
                          [0, 0, 0, 690, 478, 1, -690*500, -478*500],
                          [0, 0, 0, 5, 218, 1, -5*500, -218*500]], dtype=np.float32)

matriz_do_bem = np.array([[0],
                          [400],
                          [400],
                          [0],
                          [0],
                          [0],
                          [500],
                          [500]], dtype=np.float32)

matriz_do_mal_inv = np.linalg.inv(matriz_do_mal)

mat_mult = np.matmul(matriz_do_mal_inv, matriz_do_bem)

matriz_ultra_benefica = np.array([[mat_mult[0][0], mat_mult[1][0], mat_mult[2][0]],
                                  [mat_mult[3][0], mat_mult[4][0], mat_mult[5][0]],
                                  [mat_mult[6][0], mat_mult[7][0], 1]])


img_perspective(persp1, matriz_ultra_benefica)
img_perspective(persp2, matriz_ultra_benefica)
img_perspective(persp3, matriz_ultra_benefica)
img_perspective(persp4, matriz_ultra_benefica)

