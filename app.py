from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import mediapipe as mp


app = Flask(__name__)

modelo = tf.keras.models.load_model("modelo_landmarks_only.h5")
emociones = ['Natural', 'anger', 'fear', 'joy']

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    imagen = request.files['imagen']
    if not imagen:
        return "❌ No se envió ninguna imagen."

    img = Image.open(imagen).convert('RGB')
    img_np = np.array(img)

    resultados = face_mesh.process(img_np)

    if resultados.multi_face_landmarks:
        rostro = resultados.multi_face_landmarks[0]
        puntos = []
        for lm in rostro.landmark:
            puntos.extend([lm.x, lm.y])

        entrada = np.array(puntos).reshape(1, -1)  # (1, 936)

        pred = modelo.predict(entrada)
        emocion_idx = np.argmax(pred)
        emocion = emociones[emocion_idx]
        confianza = float(pred[0][emocion_idx]) * 100

        return f"✅ Emoción detectada: {emocion.upper()} ({confianza:.2f}%)"
    else:
        return "❌ No se detectó rostro en la imagen."

if __name__ == '__main__':
    app.run(debug=True)



# app = Flask(__name__)
# modelo = tf.keras.models.load_model("modelo_ferac_cnn_50 epocas.h5")
# emociones = ['Natural', 'anger', 'fear', 'joy']

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predecir', methods=['POST'])
# def predecir():
#     imagen = request.files['imagen']
#     img = Image.open(imagen).convert('RGB').resize((96, 96))  # Convertir a RGB y redimensionar
#     img_array = np.array(img) / 255.0
#     img_array = img_array.reshape(1, 96, 96, 3)  # Asegurar formato correcto para modelo RGB

#     prediccion = modelo.predict(img_array)
#     emocion = emociones[np.argmax(prediccion)]

#     return f"Emoción detectada: {emocion}"

# if __name__ == '__main__':
#     app.run(debug=True)

