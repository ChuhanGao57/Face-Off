import imageio
import cv2
import numpy as np 
import os,sys
import matplotlib.pyplot as plt
import tensorflow as tf
import util_recog
import util_crop
import util_attack
from util_facenet import FaceNetModel
from util_centerface import CenterFace
from l2_attack import CarliniL2
import argparse



if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_img', 
                    help='Image to be attacked')
    parser.add_argument('--self_dic', default='./aligned_imgs/matt/',
                    help='Additional aligned faces of the original person')
    parser.add_argument('--target_dic', default=None,
                    help='Target person aligned faces. None for untargeted attack')
    parser.add_argument('--model', default='center', choices=['center', 'triplet'],
                    help='Face models used for adversarial attack')
    parser.add_argument('--margin', default=0, type=float,
                    help='Margin value for CW attack')
    parser.add_argument('--target_loss', action='store_true',
                    help='Using target loss instead of default hinge loss for the attack')
    parser.add_argument('--adv_log', default='./adv_result.npz',
                    help='npz file that stores the adversarial example')
    parser.add_argument('--adv_img', default='./adv_img.png',
                    help='File name of the adversarial image')
    parser.add_argument('--amp', default=1, type=float,
                    help='Amplification factor on adversarial perturbation')
    args = parser.parse_args()

    if args.target_dic == None:
        TARGETED = False
    else:
        TARGETED = True
    
    if args.target_loss:
        HINGE = False
    else:
        HINGE = True
    
    if TARGETED:
        # Read aligned faces of targeted person
        file_list = os.listdir(args.target_dic)
        for i in range(len(file_list)):
            file_list[i] = os.path.join(args.target_dic, file_list[i])
        faces_target = util_crop.read_face_from_aligned(file_list=file_list, model=args.model)

    # Read aligned faces of the original person
    file_list = os.listdir(args.self_dic)
    for i in range(len(file_list)):
        file_list[i] = os.path.join(args.self_dic, file_list[i])
    faces_self = util_crop.read_face_from_aligned(file_list=file_list, model=args.model)

    # Read original image for the attack
    img = imageio.imread(args.attack_img)
    face, det = util_crop.crop_face(img=img, model=args.model)
    face = np.array([face])

    with tf.Session() as sess:
        if args.model == 'center':
            print("Loading Center Face Model")   
            FRmodel = CenterFace()
        elif args.model == 'triplet':
            print("Loading Triplet FaceNet Model")   
            FRmodel = FaceNetModel()
        print('Model loaded')

        # Run the attack
        adv, delta, l2, const = util_attack.find_adv(sess, 
                    face, faces_self, faces_target, FRmodel, 
                    margin=args.margin, hinge_loss=HINGE,
                    file_name=args.adv_log, model=args.model)

        #np.savez('tmp.npz',img=img, adv=adv, delta=delta, det=det, face=face)
        
            
        #dist = util_recog.face_recog(face+delta, faces_self, faces_target, FRmodel, sess)
        #print('Adv image:',dist)
        #sys.exit()

        adv_img = util_crop.apply_delta(delta=delta[0], amp=args.amp, img=img, 
                    det=det, model=args.model)
        imageio.imwrite(args.adv_img, adv_img)


        print('============= Testing adv example ================')
        adv_img = imageio.imread(args.adv_img)
        face_adv,_ = util_crop.crop_face(img=adv_img, known_det=det, model=args.model)
        face_adv = np.array([face_adv])
        dist = util_recog.face_recog(face, faces_self, faces_target, FRmodel, sess)
        print('Original image:',dist)
        dist = util_recog.face_recog(face_adv, faces_self, faces_target, FRmodel, sess)
        print('Adv image:',dist)