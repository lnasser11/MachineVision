import cv2
import numpy as np

def verificar_lata_amassada(imagem_path, limiar_threshold=60):
    imagem = cv2.imread(imagem_path)
    if imagem is None:
        print(f"Erro ao carregar {imagem_path}")
        return None

    # Escala de cinza
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Threshold simples
    _, binaria = cv2.threshold(cinza, limiar_threshold, 255, cv2.THRESH_BINARY)

    # Morfologia para limpar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binaria_limpa = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Contornos
    contornos, _ = cv2.findContours(binaria_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        print(f"{imagem_path} | Nenhum contorno encontrado.")
        return None

    maior = max(contornos, key=cv2.contourArea)
    if len(maior) < 5:
        print(f"{imagem_path} | Contorno muito pequeno para ajustar elipse.")
        return None

    # Ajuste da elipse
    elipse = cv2.fitEllipse(maior)
    (cx, cy), (w, h), angle = elipse

    # Distâncias do contorno até o centro
    distancias = [np.linalg.norm(np.array([p[0][0] - cx, p[0][1] - cy])) for p in maior]

    erros = 0
    for valor in distancias:
        if abs(valor - (h / 2)) > 2:
            erros += 1

    # Mostrar imagem com elipse
    img_resultado = imagem.copy()
    cv2.drawContours(img_resultado, [maior], -1, (0, 255, 0), 2)
    cv2.ellipse(img_resultado, elipse, (0, 0, 255), 2)
    cv2.imshow(f"Resultado - {imagem_path}", img_resultado)

    # Decisão
    amassada = erros >= 15
    status = "AMASSADA" if amassada else "OK"
    print(f"{imagem_path} | Erros: {erros} | Status: {status}")

    return amassada

# Lista de imagens
imagens = [
    "Figuras_APS3/Fig3_lata1.png",
    "Figuras_APS3/Fig3_lata2.png",
    "Figuras_APS3/Fig3_lata3.png",
    "Figuras_APS3/Fig3_lata4.png",
    "Figuras_APS3/Fig3_lata5.png"
]

# Processar todas
for imagem in imagens:
    verificar_lata_amassada(imagem)

cv2.waitKey(0)
cv2.destroyAllWindows()
