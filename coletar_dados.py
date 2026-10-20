import csv
import math
import os

import cv2
import mediapipe as mp

LETRAS = ["P", "D", "E", "O", "R"]
ARQUIVO = "dados_letras.csv"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def normalizar(mao):
    """Transforma os 21 pontos em 63 numeros relativos ao pulso e
    escalados pelo tamanho da mao (nao depende da distancia da camera)."""
    pts = [(p.x, p.y, p.z) for p in mao.landmark]
    bx, by, bz = pts[0]  # ponto 0 = pulso
    rel = [(x - bx, y - by, z - bz) for x, y, z in pts]
    escala = max(math.sqrt(x * x + y * y + z * z) for x, y, z in rel) or 1.0
    saida = []
    for x, y, z in rel:
        saida.extend([x / escala, y / escala, z / escala])
    return saida


def contar_existentes():
    contagem = {letra: 0 for letra in LETRAS}
    if os.path.exists(ARQUIVO):
        with open(ARQUIVO, newline="", encoding="utf-8") as f:
            leitor = csv.reader(f)
            next(leitor, None)  # pula cabecalho
            for linha in leitor:
                if linha and linha[0] in contagem:
                    contagem[linha[0]] += 1
    return contagem


def garantir_cabecalho():
    if not os.path.exists(ARQUIVO):
        cabecalho = ["letra"]
        for i in range(21):
            cabecalho += [f"x{i}", f"y{i}", f"z{i}"]
        with open(ARQUIVO, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(cabecalho)


garantir_cabecalho()
contagem = contar_existentes()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("ERRO: nao consegui abrir a camera. Tente trocar o 0 por 1.")
    raise SystemExit

print("Coleta iniciada. Teclas:", ", ".join(LETRAS), "| q = sair")

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERRO: nao consegui ler imagem da camera.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(rgb)

        mao = None
        if resultado.multi_hand_landmarks:
            mao = resultado.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, mao, mp_hands.HAND_CONNECTIONS)

        # Painel com as contagens
        y = 25
        for letra in LETRAS:
            cv2.putText(frame, f"{letra}: {contagem[letra]}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 28
        status = "mao detectada" if mao else "SEM MAO"
        cv2.putText(frame, status, (10, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if mao else (0, 0, 255), 2)

        cv2.imshow("Coleta de dados", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla != 255 and mao is not None:
            letra = chr(tecla).upper()
            if letra in LETRAS:
                with open(ARQUIVO, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([letra] + normalizar(mao))
                contagem[letra] += 1

cap.release()
cv2.destroyAllWindows()
print("Encerrado. Totais:", contagem)