import numpy as np
import cv2

cap = cv2.VideoCapture('C:\\Users\\lucca\\OneDrive - Insper\\Documentos\\Insper\\6\\VisMaq\\MachineVision\\Projetos\\APS1\\Figuras_APS1\\Video_APS1_3.avi')

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Número de frames:', num_frames)

ret, primeiro_frame = cap.read()
primeiro_frame_gray = cv2.cvtColor(primeiro_frame, cv2.COLOR_BGR2GRAY)

frame_atual = 1

while(frame_atual < num_frames):
    ret, frame = cap.read()
    
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = frame_gray.astype(np.int32) - primeiro_frame_gray.astype(np.int32)
    diff = np.abs(diff).astype(np.uint8)

    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    contornos = np.where(thresh > 0, 255, 0).astype(np.uint8)

    scale_percent = 30
    width = int(frame_gray.shape[1] * scale_percent / 100)
    height = int(frame_gray.shape[0] * scale_percent / 100)
    tamanho = (width, height)

    resized_carros = cv2.resize(frame, tamanho, interpolation=cv2.INTER_AREA)
    resized_contornos = cv2.resize(contornos, tamanho, interpolation=cv2.INTER_AREA)

    cv2.imshow('Carros', resized_carros)
    cv2.imshow('Contornos Carros', resized_contornos)

    frame_atual += 1

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
