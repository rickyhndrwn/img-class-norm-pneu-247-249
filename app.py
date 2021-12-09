from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('./static/ml_model/cxr_model.h5')

def predict_label(img_path):
    img = load_img(img_path, target_size=(180,180))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    pred_result = model.predict(img)

    pred_value = (pred_result[0][0] > 0.5).astype(np.int)
    pred_label = 'Normal' if pred_value == 0 else 'Pneumonia'

    return pred_label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path)
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction)

    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run()