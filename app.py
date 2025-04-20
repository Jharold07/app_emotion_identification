from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import mediapipe as mp
from tensorflow.keras.applications.mobilenet import preprocess_input


app = Flask(__name__)

modelo = tf.keras.models.load_model("emotion_face_mobilNet.h5")
emociones = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    imagen = request.files['imagen']
    if not imagen:
        return render_template('index.html', error="❌ No se envió ninguna imagen.")

    # Preprocesamiento de imagen
    img = Image.open(imagen).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    # Predicción
    pred = modelo.predict(img_array)
    emocion_idx = np.argmax(pred)
    emocion = emociones[emocion_idx]
    confianza = float(pred[0][emocion_idx]) * 100

    #return render_template('index.html', emocion=emocion.upper(), confianza=f"{confianza:.2f}%")
    return f"Emoción detectada: {emocion}, Confianza: {confianza:.2f}%"

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

