from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import numpy as np
from keras.models import load_model
from keras.applications.mobilenet import MobileNet


from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
from keras.models import load_model
from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array, load_img
import os
import tensorflow as tf

# Mobile Net


# モデルの保存、モデルをロードした後予測しかしないため、include_optimizer=Falseとする
#model.save('test.h5', include_optimizer=False)

# Define a flask app
app = Flask(__name__)


# Model saved with Keras model.save()
MODEL_PATH = 'classifier_image.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

#model = MobileNet(weights="imagenet", include_top=True)
def load_mobilenet_model():
    global model
    model = load_model('classifier_image.h5')
    global graph
graph = tf.get_default_graph()



# Model saved with Keras model.save()
#MODEL_PATH = 'models/your_model.h5'

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        with graph.as_default():
            preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = preds.argmax(axis=0)   # ImageNet Decode
        result = str(pred_class[0])               # Convert to string
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()