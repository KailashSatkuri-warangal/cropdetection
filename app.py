from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'leaf_image' not in request.files:
        return redirect(request.url)
    file = request.files['leaf_image']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        prediction, confidence = predict_leaf(filepath)
        
        if prediction == 'invalid':
            message = "Please input a valid leaf image."
        else:
            message = f"The leaf is {prediction} with a confidence of {confidence:.2f}%."
        
        return render_template('result.html', message=message, filename=filename)
    else:
        return render_template('index.html', message="Please upload a valid image.")
def predict_leaf(image_path):
    model = load_model('models/crop_infection_model.keras')
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  #you get batch
    img_array /= 255.0  #Normalise the image
    prediction = model.predict(img_array)
    pred_value = prediction[0][0]  
    if pred_value < 0.45:  #for healthy leaves
        return 'healthy', (1 - pred_value) * 100
    elif pred_value > 0.55:  #infected leaves
        return 'infected', (1 - pred_value) * 100
    else:
        return 'uncertain or undected able please Upload leaf image', (1 - pred_value) * 100  
if __name__ == '__main__':
    app.run(debug=True)
