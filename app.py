from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os, os.path
import numpy as np
import pickle
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import cv2
import matplotlib
import keras
import shutil

import tensorflow as tf

from tensorflow.python.keras.backend import set_session
from threading import Timer

from flask import make_response

import random
import string


app = Flask(__name__,static_folder=os.path.abspath('static'))

#thread1 load model
session = tf.Session(graph=tf.Graph())
with session.graph.as_default():
	keras.backend.set_session(session)
	model = pickle.load(open('unet_model.pkl','rb'))
	


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD = 'static/'
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def delete_images():
  dirpath = os.path.join(app.config['UPLOAD_FOLDER'])
  for filename in os.listdir(dirpath):
      if filename.endswith('.gitkeep') !=True:
        filepath = os.path.join(dirpath, filename)
        try:
          shutil.rmtree(filepath)
        except OSError:
          os.remove(filepath)

def generate_noisefree_image(image_dir):
    default_image_size = (48, 48)
    
    image = cv2.imread(image_dir)
    image = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY)
    
    image = cv2.resize(image, default_image_size)
    img_array = img_to_array(image)
    
    x_test = img_array.astype('float32') / 255.0
    x_test = np.clip(x_test, 0., 1.)
    
    x_enpanded = np.expand_dims(x_test, axis=0)
    
    #Thread 2 model prediction
    with session.graph.as_default():
    	keras.backend.set_session(session)
    	generated_image = model.predict(x_enpanded)
    
    return generated_image

def delete_images():
    folder = os.path.join(app.config['UPLOAD_FOLDER'])
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            shutil.rmtree(filepath)
        except:
            os.remove(filepath)
                    

@app.route('/')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        image_dir = (os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        
        generated_image = generate_noisefree_image(image_dir)
        generated_image = generated_image.reshape((48,48))
        
        file_name_image = ''.join(random.choice(string.ascii_lowercase) for i in range(16))
        
        matplotlib.image.imsave(os.path.join(app.config['UPLOAD_FOLDER'])+file_name_image+'.png', generated_image)
        images_names = os.listdir('static')
        
        for names in images_names:
            response = make_response(render_template('result.html', image_names = images_names))
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            t = Timer(2,delete_images)
            t.start()
            
            return response
        

if __name__ == '__main__':
	app.run(debug = True)
