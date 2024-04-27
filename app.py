from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import keras
import numpy as np

app = Flask(__name__)
model = keras.models.load_model('model/kidney_stone_detection_model.h5')

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/kidney_model', methods=['GET'])
def kdm():
    return render_template('model.html')

@app.route('/kidney_model', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(150, 150, 3))
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    image = image/255.0
    # Make predictions
    prediction = model.predict(image)
    if prediction < 0.5:
        output = 'Image is Normal'
    else:
        output = 'Image have a Kidney Stone'
    return render_template('model.html', output=output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)