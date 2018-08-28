import sys
import os
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import scipy.misc as ms
import imageio
import cv2

def pre_proc(img, model):
    # img: read in by imageio.imread
    # with shape (x,y,3), in the format of RGB
    if model == 'center':
        # convert to (3,112,96) with BGR
        img_resize = ms.imresize(img, (112, 96), interp='bilinear')
        img_BGR = img_resize[...,::-1]
        img_CHW = (img_BGR.transpose(2, 0, 1) - 127.5) / 128
        return img_CHW
    elif model == 'triplet':
        img_resize = ms.imresize(img, (96, 96), interp='bilinear')
        img_CHW = np.around(np.transpose(img_resize, (2,0,1))/255.0, decimals=12)
        return img_CHW



def crop_face(img, known_det=None, model='triplet'):
    #img: (x,y,3) in range [0,255], RGB
    #face: (3,96,96), RGB or GBR depending on model

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    margin = 4
    image_width = 96
    image_height = 96
    if model == 'center':
        image_height = 112

    if(known_det is not None):
        det = known_det
        print('Using a given boudning box')
        img_size = np.asarray(img.shape)[0:2]
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(known_det[0]-margin/2, 0)
        bb[1] = np.maximum(known_det[1]-margin/2, 0)
        bb[2] = np.minimum(known_det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(known_det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        scaled = ms.imresize(cropped, (image_height, image_width), interp='bilinear')
    else:
        print('Trying to find a bounding box')
        try: 
            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
        except:
            print('Error detecting')
            return 

        if nrof_faces != 1:
            print('Error, found {} faces'.format(nrof_faces))
            return

        
        det = bounding_boxes[:,0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        det_arr.append(np.squeeze(det))

        for i, det in enumerate(det_arr):
            
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            scaled = ms.imresize(cropped, (image_height, image_width), interp='bilinear')

    if model == 'center':
        scaled = scaled[...,::-1]
    
    face = np.around(np.transpose(scaled, (2,0,1))/255.0, decimals=12)
    
    if model == 'center':
        face = (face-0.5)*2
    
    face = np.array(face)    
    
    return face, det


def apply_delta(delta, amp, img, det, model):
    # model = 'triplet'
        # img: (x,y,3) in range [0,255], RGB
        # delta: (3,96,96), RGB
        # adv_img: (x,y,3) in range [0,1], RGB
    # model = 'center'
        # img: (x,y,3) in range [0,255], RGB
        # delta: (3,96,96), BGR
        # adv_img: (x,y,3) in range [0,1], RGB
    


    if(np.max(img, axis=(0,1,2)) <= 1):
        print('Error: face image pixel should be in range [0,255]')
    adv_img = img / 255.0
    
    margin = 4
    img_size = np.asarray(img.shape)[0:2]
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])

    orig_dim = [bb[3]-bb[1], bb[2]-bb[0]]

    if model == 'center':
        delta = delta[::-1,...]
    delta = np.transpose(delta, (1,2,0))
    delta = delta * amp
    delta_up = cv2.resize(delta, (orig_dim[1], orig_dim[0]))
    adv_img[bb[1]:bb[3],bb[0]:bb[2],:] += delta_up
    adv_img = np.maximum(adv_img, 0)
    adv_img = np.minimum(adv_img, 1)

    return adv_img


def read_face_from_files(file_list, model):
    result = []
    for file_name in file_list:
        print(file_name)
        img = imageio.imread(file_name)

        face, _ = crop_face(img)
        result.append(face)
    result = np.array(result)
    return result

def read_face_from_aligned(file_list, model):
    result = []
    for file_name in file_list:
        print(file_name)
        face = imageio.imread(file_name)
        face = pre_proc(face, model=model)
        result.append(face)
    result = np.array(result)
    return result

def center_to_triplet(img):
    #img: (3,112,96), BGR, pixel in [-1,1]
    img = (img+1)/2
    face_RGB = img[::-1,...]
    face_HWC = np.transpose(face_RGB,(1,2,0))
    face_96 = ms.imresize(face_HWC, (96, 96), interp='bilinear')
    #plt.imshow(face_96)
    #plt.show()
    face_96_CHW = np.transpose(face_96,(2,0,1))
    #face_96_CHW = np.array([face_96_CHW])
    face_96_CHW = face_96_CHW / 255
    return face_96_CHW

def triplet_to_center(img):
    #img: (3,96,96), RGB, pixel in [0,1]
    img_HWC = np.transpose(img,(1,2,0))
    img_112 = ms.imresize(img_HWC, (112, 96), interp='bilinear')
    img_CHW = np.transpose(img_112,(2,0,1))
    #img_CHW = np.array([img_CHW])
    img_CHW = (img_CHW - 127.5) / 128
    return img_CHW
