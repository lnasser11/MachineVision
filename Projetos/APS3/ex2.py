import cv2
import numpy as np


def detectar_pecas_info(imagem_path):
    imagem = cv2.imread(imagem_path)
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    altura_img, largura_img = imagem.shape[:2]

    faixa_esteira_y_min = int(0.25 * altura_img)
    faixa_esteira_y_max = int(0.85 * altura_img)

    azul_mask = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
    verde_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))

    resultados = []

    # Configuração do detector de blobs
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 100
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.blobColor = 255

    params2 = cv2.SimpleBlobDetector_Params()
    params2.filterByArea = True
    params2.minArea = 100
    params2.maxArea = 2000
    params2.filterByCircularity = False
    params2.filterByConvexity = False
    params2.filterByInertia = False
    params2.blobColor = 255

    detector = cv2.SimpleBlobDetector_create(params)
    detector2 = cv2.SimpleBlobDetector_create(params2)

    for cor_mask, cor_nome in [(azul_mask, "azul"), (verde_mask, "verde")]:
        contornos, _ = cv2.findContours(cor_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area < 2000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            centro_x = x + w // 2
            centro_y = y + h // 2

            if not (faixa_esteira_y_min < centro_y < faixa_esteira_y_max):
                continue

            roi = imagem[y:y + h, x:x + w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Pré-processamento para etiquetas brancas
            _, bin_thresh = cv2.threshold(roi_gray, 200, 255,
                                          cv2.THRESH_BINARY)

            # Detectar blobs
            keypoints = detector.detect(bin_thresh)
            keypoints2 = detector2.detect(bin_thresh)
            num_etiquetas = len(keypoints) + len(keypoints2) * 2

            status = "Aprovado" if num_etiquetas >= 3 else "Reprovado"

            resultados.append({
                "cor": cor_nome,
                "posicao": (centro_x, centro_y),
                "etiquetas": num_etiquetas,
                "status": status
            })

            # Visualização
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(imagem, f"{cor_nome}, {status}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Debug: mostrar blobs detectados
            roi_com_blobs = cv2.drawKeypoints(
                roi, keypoints, np.array([]), (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            roi_com_blobs2 = cv2.drawKeypoints(
                roi_com_blobs, keypoints2, np.array([]), (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow(f"Blobs detectados ({cor_nome})", roi_com_blobs2)

    # Peça mais à direita
    if resultados:
        direita = max(resultados, key=lambda r: r["posicao"][0])
        print("🔍 Peça mais à direita (NA ESTEIRA):")
        print(f"➡️ Cor: {direita['cor']}")
        print(f"📍 Posição (X,Y): {direita['posicao']}")
        print(f"📦 Etiquetas: {direita['etiquetas']}")
        print(f"✅ Status: {direita['status']}")
    else:
        print("Nenhuma peça válida detectada na esteira.")

    cv2.imshow("Resultado Final", imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


paths = [
    r"Figuras_APS3\Fig2_Esteira1.png", 
    r"Figuras_APS3\Fig2_Esteira2.png",
    r"Figuras_APS3\Fig2_Esteira3.png"
]

for img in paths:
    detectar_pecas_info(img)