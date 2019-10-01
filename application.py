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
import tensorflow as tf



from flask import Flask, render_template, request, redirect, url_for
import stripe
import os
import base64
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask

# Define a flask app
application = Flask(__name__)

import pyrebase
config = {
	"apiKey": "AIzaSyAUc2GZi4oA22bjz1Gcw1OLIQWAgGapAXE",
    "authDomain": "imageclassifier-712c4.firebaseapp.com",
    "databaseURL": "https://imageclassifier-712c4.firebaseio.com",
    "projectId": "imageclassifier-712c4",
    "storageBucket": "",
    "messagingSenderId": "221619179133",
    "appId": "1:221619179133:web:14d777dae04f9d05a32b14"
}
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

@application.route('/')
@application.route('/index', methods=['GET', 'POST'])
def index():
    if (request.method == 'POST'):
            email = request.form['name']
            password = request.form['password']
            try:
                auth.sign_in_with_email_and_password(email, password)
                #user_id = auth.get_account_info(user['idToken'])
                #session['usr'] = user_id
                return redirect(url_for('payment'))
            except:
                unsuccessful = 'Please check your credentials'
                return render_template('index.html', umessage=unsuccessful)
    return render_template('index.html')

@application.route('/create_account', methods=['GET', 'POST'])
def create_account():
    if (request.method == 'POST'):
            email = request.form['name']
            password = request.form['password']
            auth.create_user_with_email_and_password(email, password)
            return render_template('index.html')
    return render_template('create_account.html')

@application.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if (request.method == 'POST'):
            email = request.form['name']
            auth.send_password_reset_email(email)
            return render_template('index.html')
    return render_template('forgot_password.html')


pub_key = 'pk_test_NTpznax6Yy3eHdruwGFRbHSY00fNgoujU5'
secret_key = 'sk_test_ekHEsZ0D6wMLcJGkenE3PORZ00MGL1bZSp'

stripe.api_key = secret_key
@application.route('/payment')
def payment():
    return render_template('payment.html', pub_key=pub_key)

@application.route('/thanks')
def thanks():
    return render_template('thanks.html')

@application.route('/predict')
def predict():
    return render_template('/home.html')

@application.route('/pay', methods=['POST'])
def pay():
    
    customer = stripe.Customer.create(email=request.form['stripeEmail'], source=request.form['stripeToken'])

    charge = stripe.Charge.create(
        customer=customer.id,
        amount=2000,
        currency='usd',
        description='The Product'
    )

    return redirect(url_for('predict'))


# Model saved with Keras model.save()
MODEL_PATH = 'classifier_image.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')
def load_mobilenet_model():
    global model
    model = load_model('classifier_image.h5')
    global graph
graph = tf.get_default_graph()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


#@app.route('/', methods=['GET'])
#def index():
    # Main page
    #return render_template('index.html')


@application.route('/predict', methods=['GET', 'POST'])
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
        pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0])
        return result

        
        
    return None



if __name__ == '__main__':
    # app.run(port=5002, debug=True)
    application.run(port=5002, debug=True)

    # Serve the app with gevent
    #
    #http_server = WSGIServer(('0.0.0.0', 5000), application)
    #http_server.serve_forever()