from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

app = Flask(__name__)
modelo = tf.keras.models.load_model("modelo_emociones.h5")
emociones = ['Enojado', 'Feliz', 'Triste', 'Sorprendido', 'Neutral']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    imagen = request.files['imagen']
    img = Image.open(imagen).convert('L').resize((48, 48))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)

    prediccion = modelo.predict(img_array)
    emocion = emociones[np.argmax(prediccion)]

    return f"Emoción detectada: {emocion}"

if __name__ == '__main__':
    app.run(debug=True)
