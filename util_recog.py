import scipy.misc as ms 
import imageio
import numpy as np 
import os,sys
from util_crop import *
import matplotlib.pyplot as plt
import tensorflow as tf
from util_recog import read_face_from_files
import tensorflow as tf
from util_facenet import FaceNetModel
import cv2
from keras.models import load_model



def face_recog(face, face_stack_1, face_stack_2, FRmodel, sess):
    # face: (3,96,96), RGB
    input_tensor = tf.placeholder(tf.float32, (None, 3, 96, 96))
    embedding = FRmodel.predict(input_tensor)
    
    encoding_face = sess.run(embedding, feed_dict={input_tensor: face})
    encoding_p1 = sess.run(embedding, feed_dict={input_tensor: face_stack_1})
    encoding_p2 = sess.run(embedding, feed_dict={input_tensor: face_stack_2})
    dist_mtx = np.zeros((2,encoding_p1.shape[0]))
    for i in range(encoding_p1.shape[0]):
        dist_p1 = np.linalg.norm(encoding_face - encoding_p1[i:i+1])
        dist_p2 = np.linalg.norm(encoding_face - encoding_p2[i:i+1])
        dist_mtx[0,i] = dist_p1
        dist_mtx[1,i] = dist_p2
    
    return np.mean(dist_mtx,axis=1) 

def face_recog_center(face, face_stack_1, face_stack_2, center_model, sess=None):
    # face: (3,112,96), BGR
    encoding_face = center_model.model.predict(face)
    encoding_p1 = center_model.model.predict(face_stack_1)
    encoding_p2 = center_model.model.predict(face_stack_2)
    dist_mtx = np.zeros((2,encoding_p1.shape[0]))
    for i in range(encoding_p1.shape[0]):
        dist_p1 = np.linalg.norm(encoding_face - encoding_p1[i:i+1])
        dist_p2 = np.linalg.norm(encoding_face - encoding_p2[i:i+1])
        dist_mtx[0,i] = dist_p1
        dist_mtx[1,i] = dist_p2
    
    return np.mean(dist_mtx,axis=1) 


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    model = 'center'

    file_list = os.listdir('./aligned_imgs/leo/')
    for i in range(len(file_list)):
        file_list[i] = os.path.join('./aligned_imgs/leo/',file_list[i])
    faces_leo = read_face_from_aligned(file_list,model=model)

    file_list = os.listdir('./aligned_imgs/matt/')
    for i in range(len(file_list)):
        file_list[i] = os.path.join('./aligned_imgs/matt/',file_list[i])
    faces_matt = read_face_from_aligned(file_list,model=model)

    
    face = imageio.imread('./aligned_imgs/matt/matt_02.jpg')
    face = pre_proc(face, model=model)
    face = np.array([face])

    keras_model = load_model('model_conver/face_model_caffe_converted.h5')
    dist = face_recog_center(face, faces_matt, faces_leo, keras_model)
    print(dist)


    # print(face.shape)
    # #print(face_recog_center(face, faces_matt, faces_leo, center_model))

    # with tf.Session() as sess:
    #     print("Loading FaceNet Model")   
    #     FRmodel = FaceNetModel()
    #     print('Model loaded')
    #     dist = face_recog(face, faces_matt, faces_leo, FRmodel, sess)
    #     print('completed')
    #     print(dist)


    





