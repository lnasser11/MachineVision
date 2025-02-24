import numpy as np
import cv2

cap = cv2.VideoCapture('C:\\Users\\lucca\\OneDrive - Insper\\Documentos\\Insper\\6\\VisMaq\\MachineVision\\Projetos\\APS1\\Figuras_APS1\\Video_APS1_3.avi')

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Número de frames:', num_frames)

frame_atual = 1
ret, frame_anterior = cap.read()

frame_anterior_gray = cv2.cvtColor(frame_anterior, cv2.COLOR_BGR2GRAY)

while(frame_atual < num_frames):
    ret, frame = cap.read()
    
    if not ret:
        print("Não foi possível ler o frame.")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(frame_gray, frame_anterior_gray)

    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    contornos = np.where(thresh > 0, 255, 0).astype(np.uint8)

    scale_percent = 30
    width = int(frame_gray.shape[1] * scale_percent / 100)
    height = int(frame_gray.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_gray = cv2.resize(frame_gray, dim, interpolation=cv2.INTER_AREA)
    resized_contornos = cv2.resize(contornos, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('Video Cinza', resized_gray)
    cv2.imshow('Contornos Carros', resized_contornos)

    frame_anterior_gray = frame_gray.copy()
    frame_atual += 1

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
