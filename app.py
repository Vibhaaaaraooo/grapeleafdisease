from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the trained model
model = tf.keras.models.load_model("grape_leaf_cnn_model.h5")

# Define class labels
class_labels = ['Healthy', 'Powdery', 'Rust']

def preprocess_image(image_path):
    """Preprocess image for model prediction."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Expand dimensions for model input
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        image = preprocess_image(file_path)
        prediction = model.predict(image)
        predicted_class = class_labels[np.argmax(prediction)]
        
        return render_template('index.html', uploaded_image=file_path, prediction=predicted_class)
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
