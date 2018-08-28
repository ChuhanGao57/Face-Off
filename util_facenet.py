import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from keras.models import load_model
from keras.models import model_from_json
import time
import keras


def triplet_loss(y_true, y_pred, alpha = 0.3):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

class FaceNetModel:
    def __init__(self, session=None):
        self.num_channels = 3
        self.image_height = 96
        self.image_width = 96


        
        json_file = open('FRmodel.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        FRmodel = model_from_json(loaded_model_json)
        # load weights into new model
        FRmodel.load_weights("FRmodel.h5")
        #FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
        print("Loaded model from disk")
        self.model = FRmodel
    
    def predict(self, im):
        # return self.model.predict_on_batch(im)
        return self.model(im)
        # return tf.reduce_sum(im, 0)
