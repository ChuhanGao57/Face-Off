import scipy.misc as ms 
import imageio
import cv2
import numpy as np 
import os,sys
import matplotlib.pyplot as plt
import tensorflow as tf
from util_recog import *
from util_crop import *
from util_facenet import FaceNetModel
from util_centerface import CenterFace
from l2_attack import CarliniL2


# ATTENTION!
# 'adv' and 'delta' are RGB for triplet model, BGR for center face model

def find_adv(sess, face, face_stack_self, face_stack_target, FRmodel, file_name=None, margin=0, hinge_loss=True, model='triplet'):
    const_high = 10.0
    const_low = 0.05
    const = 0.3
    ever_success = False
    best_l2 = 9999.0
    best_adv = None
    best_delta = None
    best_const = None
    

    batch_size = face.shape[0]
    self_size = face_stack_self.shape[0]
    target_size = face_stack_target.shape[0]

    for ii in range(5):
        print("Search #",ii,"with constant",const)
        if model == 'center':
            boxmin = -1
            boxmax = 1
        if model == 'triplet':
            boxmin = 0
            boxmax = 1
        attack = CarliniL2(sess, FRmodel, batch_size=batch_size,
                    learning_rate=0.01,hinge_loss=hinge_loss ,targeted=True,
                    self_db_size=self_size,target_batch_size=target_size,
                    initial_const=const, max_iterations=500, confidence=margin,
                    boxmin=boxmin, boxmax=boxmax)
        adv, delta = attack.attack(face, face_stack_target, face_stack_self)
        if model == 'triplet': 
            dist = face_recog(adv, face_stack_self, face_stack_target, FRmodel, sess)
        if model == 'center':
            dist = face_recog_center(adv, face_stack_self, face_stack_target, FRmodel, sess)
            print(dist)
        if(dist[0] - dist[1] >= margin):
            # Successfully found adv example
            print('Success with const',const)
            ever_success = True
            adv_l2 = np.linalg.norm(delta)
            if(adv_l2) < best_l2:
                best_l2 = adv_l2
                best_adv = adv
                best_delta = delta
                best_const = const
            # decrease const
            const_high = const
            const = (const_high + const_low) / 2
        else:
            # Faild to find adv example
            print('Failure with const',const)
            const_low = const
            const = (const_high + const_low) / 2
        if(ever_success == True and const_high-const_low < 0.02):
            break
    
    if(ever_success):
        print('Successfully found adv example')
    else:
        print('Failed to find adv example')
    
    if(file_name):
        np.savez(file_name, face=face, adv=best_adv, delta=best_delta, l2=best_l2)

    return best_adv, best_delta, best_l2, best_const


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

    
    face = imageio.imread('./aligned_imgs/matt/matt_03.jpg')
    face = pre_proc(face, model=model)
    face = np.array([face])

    with tf.Session() as sess:
        if model == 'triplet':
            print("Loading FaceNet Model")   
            FRmodel = FaceNetModel()
            print('Model loaded')
        if model == 'center':
            print("Loading CenterFace Model")   
            FRmodel = CenterFace()
            print('Model loaded')

        adv, delta, l2, const = find_adv(sess, 
                    face, faces_matt, faces_leo, FRmodel, 
                    margin=10, hinge_loss=True,
                    file_name='./adv_imgs/test.npy',
                    model='center')


    








