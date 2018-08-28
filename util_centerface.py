import os,sys
from keras.models import load_model


class CenterFace:
    def __init__(self, session=None):
        self.num_channels = 3
        self.image_height = 112
        self.image_width = 96
        self.model = load_model('model_conver/face_model_caffe_converted.h5')
    
    def predict(self, im):
        # return self.model.predict_on_batch(im)
        return self.model(im)
        # return tf.reduce_sum(im, 0)