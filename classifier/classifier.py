import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
# Model	      Size	Top-1  	Top-5 	Parameters	Depth	Time (ms)(CPU)	Time (ms) (GPU)
# VGG16	  ==> 528	71.3%	90.1%	138.4M	    16	    69.5	        4.2
# ResNet50  ==> 98	74.9%	92.1%	25.6M	    107	    58.2	        4.6
# MobileNet ==> 16    70.4%	89.5%	4.3M	    55	    22.6	        3.4


def predict_density(image):

    with tf.device('/cpu:0'):
        model = load_model('./classifier/resnet50.h5')
        classes = ['Dense', 'Dense', "Not Dense"]

        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = tf.convert_to_tensor(image)
        
        result = model.predict(image, batch_size=1)
        result = np.argmax(result, axis=1)
        
        return classes[result[0]]