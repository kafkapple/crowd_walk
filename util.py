# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 20:04:34 2018

@author: 2014_Joon_IBS
"""

import cv2
import dlib
import os
from imutils import face_utils
landmarks = os.path.join(os.getcwd(),'src','shape_predictor_68_face_landmarks.dat')
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmarks)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

face_shape = (48, 48)
RED_COLOR = (0, 0, 255)
WHITE_COLOR = (255, 255, 255)
pink = (255,139,148)
green = (255, 170, 165) # 하늘색? 에메랄드 그린
dahong = (254,74,173)#(241, 153, 160)


def convert_dlib(picture):
    gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image, 1) #Detect the faces in the image
    for k,d in enumerate(detections): #For each detected face
        shape = predictor(clahe_image, d) #Get coordinates
        for i in range(1,68): # To draw 68 landmarks from dlib
            cv2.circle(picture, (shape.part(i).x, shape.part(i).y), 1, (0,0,200), thickness=1) 
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            #For each point, draw a red circle with thickness2 on the original frame
            
        for x, y in zip(xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.
            landmarks.append(x)
            landmarks.append(y)

    #cv2.imwrite(os.path.join(path,class_label[labels[i]], img_name), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    face = crop_face(img, face_coordinates)
    if face is not None:
        face_resize = cv2.resize(face, face_shape)
        face_gray = cv2.cvtColor(face_resize, cv2.COLOR_BGR2GRAY)
        #face_gray = clahe.apply(face_gray)/ 255.  # histogram equalization & normalize (0~1)
        
    cv2.imwrite( picture, [cv2.IMWRITE_PNG_COMPRESSION, 0])
 
    
    face_coordinates = dlib_face_coordinates(frame) 
#    
#def drawFace(frame, face_coordinates):
#    if face_coordinates is not None:
#        (x, y, w, h) = face_coordinates
#        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=1)


def check_resize_area(face_coordinates):
    (x, y, w, h) = face_coordinates
    if x - int(w / 4) > 0 and y - int(h / 4) > 0:
        return True
    return False

def crop_face(frame, face_coordinates):
    cropped_img = frame
    (x, y, w, h) = face_coordinates
    if check_resize_area(face_coordinates):
        cropped_img = frame[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
        #cv2.imwrite('./face.jpg', cropped_img, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
        return cropped_img
    else:
        return None