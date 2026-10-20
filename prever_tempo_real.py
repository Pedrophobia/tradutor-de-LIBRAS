import math

import cv2
import joblib
import mediapipe as mp
import pandas as pd

ARQUIVO_MODELO = "modelo_rf.joblib"
LIMIAR = 0.80          # confianca minima para aceitar a letra
FRAMES_ESTAVEIS = 15   # quantos frames seguidos a mesma letra precisa aparecer

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

modelo = joblib.load(ARQUIVO_MODELO)

# Mesmos nomes de colunas usados no treino (x0,y0,z0,...,x20,y20,z20)
COLUNAS = []
for i in range(21):
    COLUNAS += [f"x{i}", f"y{i}", f"z{i}"]


def normalizar(mao):
    pts = [(p.x, p.y, p.z) for p in mao.landmark]
    bx, by, bz = pts[0]
    rel = [(x - bx, y - by, z - bz) for x, y, z in pts]
    escala = max(math.sqrt(x * x + y * y + z * z) for x, y, z in rel) or 1.0
    saida = []
    for x, y, z in rel:
        saida.extend([x / escala, y / escala, z / escala])
    return saida


def texto(frame, msg, pos, cor=(0, 255, 0), tam=0.9):
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, tam, cor, 2)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("ERRO: nao consegui abrir a camera. Tente trocar o 0 por 1.")
    raise SystemExit

print("Teclas: c = limpar texto | b = apagar ultima letra | q = sair")

frase = []
letra_candidata = None
contador = 0
ja_adicionada = False

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

        letra_atual = None
        confianca = 0.0

        if resultado.multi_hand_landmarks:
            mao = resultado.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, mao, mp_hands.HAND_CONNECTIONS)

            entrada = pd.DataFrame([normalizar(mao)], columns=COLUNAS)
            probs = modelo.predict_proba(entrada)[0]
            idx = probs.argmax()
            confianca = probs[idx]
            if confianca >= LIMIAR:
                letra_atual = modelo.classes_[idx]

        # Regra de estabilidade: a letra so entra na frase se aparecer
        # por varios frames seguidos, e nao repete ate mudar ou tirar a mao.
        if letra_atual is not None and letra_atual == letra_candidata:
            contador += 1
        else:
            letra_candidata = letra_atual
            contador = 1 if letra_atual is not None else 0
            ja_adicionada = False

        if (letra_candidata is not None and contador >= FRAMES_ESTAVEIS
                and not ja_adicionada):
            frase.append(letra_candidata)
            ja_adicionada = True

        # Interface
        if letra_atual:
            texto(frame, f"{letra_atual}  ({confianca:.0%})", (10, 40))
        elif resultado.multi_hand_landmarks:
            texto(frame, "incerto...", (10, 40), (0, 165, 255))
        else:
            texto(frame, "sem mao", (10, 40), (0, 0, 255))

        texto(frame, "Texto: " + "".join(frase), (10, frame.shape[0] - 20),
              (255, 255, 255))

        cv2.imshow("Reconhecimento em tempo real", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        elif tecla == ord("c"):
            frase = []
        elif tecla == ord("b") and frase:
            frase.pop()

cap.release()
cv2.destroyAllWindows()
print("Texto final:", "".join(frase))