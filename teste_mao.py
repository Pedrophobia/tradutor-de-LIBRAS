import cv2
import mediapipe as mp

print("Iniciando...")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERRO: não consegui abrir a câmera. Tente trocar o 0 por 1.")
    raise SystemExit

print("Câmera aberta. Aperte q na janela para sair.")

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERRO: não consegui ler imagem da câmera.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(rgb)

        if resultado.multi_hand_landmarks:
            for mao in resultado.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, mao, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Teste MediaPipe", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("Encerrado.")