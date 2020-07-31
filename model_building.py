# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2
import matplotlib


model = pickle.load(open('unet_model.pkl','rb'))

def generate_noisefree_image(image_dir):
    default_image_size = (48, 48)
    
    image = cv2.imread(image_dir)
    image = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY)
    
    image = cv2.resize(image, default_image_size)
    img_array = img_to_array(image)
    
    x_test = img_array.astype('float32') / 255.0
    x_test = np.clip(x_test, 0., 1.)
    
    x_enpanded = np.expand_dims(x_test, axis=0)
    
    generated_image = model.predict(x_enpanded)
    
    return generated_image

generated_image = generate_noisefree_image('test_image.png')

generated_image = generated_image.reshape((48,48))

matplotlib.image.imsave('generated_image.png', generated_image)