# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:31:59 2018

@author: 2014_Joon_IBS
"""

import argparse
import sys
import numpy as np

import os
import glob
import time
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl

from imutils.video import VideoStream
from scipy.misc import imsave

import matplotlib.animation as animation
from matplotlib import style
from io import StringIO

from scipy import signal

import cv2
import dlib
from imutils import face_utils

import time
import sys

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
angry_check = 0
# dlib을 위한 변수
landmarks = os.path.join(os.getcwd(),'src','shape_predictor_68_face_landmarks.dat')
#landmarks = './src/shape_predictor_68_face_landmarks.dat'  # jj_modify for relative path to the dat
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmarks)

FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
RED_COLOR = (0, 0, 255)
WHITE_COLOR = (255, 255, 255)
pink = (255,139,148)
green = (255, 170, 165) # 하늘색? 에메랄드 그린
dahong = (254,74,173)#(241, 153, 160)


cur_color = pink#dahong#RED_COLOR#(255, 170, 165) # 에메랄드 그린
color = dahong #pink

cur_font = cv2.FONT_HERSHEY_DUPLEX

cam_width, cam_height = 0, 0
expand_width, expand_height = 0, 0
reduce_width, reduce_height = 0, 0
min_x, max_x, min_y, max_y = 0, 0, 0, 0

#(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
#(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

## Face

def dlib_face_coordinates(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    return detector(gray, 0)

def drawFace(frame, face_coordinates):
    if face_coordinates is not None:
        (x, y, w, h) = face_coordinates
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=1)

def crop_face(frame, face_coordinates):
    cropped_img = frame
    (x, y, w, h) = face_coordinates
    if check_resize_area(face_coordinates):
        cropped_img = frame[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
        #cv2.imwrite('./face.jpg', cropped_img, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
        return cropped_img
    else:
        return None

def check_resize_area(face_coordinates):
    (x, y, w, h) = face_coordinates
    if x - int(w / 4) > 0 and y - int(h / 4) > 0:
        return True
    return False


def preprocess(img, face_coordinates, face_shape=(48, 48)):
    face = crop_face(img, face_coordinates)
    if face is not None:
        face_resize = cv2.resize(face, face_shape)
        face_gray = cv2.cvtColor(face_resize, cv2.COLOR_BGR2GRAY)
        #face_gray = clahe.apply(face_gray)/ 255.  # histogram equalization & normalize (0~1)
   
        return face_resize
    else:
        return None


def set_default_min_max_area(width, height):
    global cam_width, cam_height, min_x, max_x, min_y, max_y, \
        expand_width, expand_height, reduce_width, reduce_height
    cam_width, cam_height = width, height
    min_x = int(width * 0.1)
    max_x = int(width * 0.9)
    min_y = int(height * 0.1)
    max_y = int(height * 0.9)
    expand_width = int(cam_width * 0.05)
    expand_height = int(cam_height * 0.05)
    reduce_width = int(cam_width * 0.05)
    reduce_height = int(cam_height * 0.05)


def expend_detect_area():
    global min_x, max_x, min_y, max_y
    min_x -= expand_width
    min_y -= expand_height
    max_x += expand_width
    max_y += expand_height


def reduce_detect_area():
    global min_x, max_x, min_y, max_y
    min_x += reduce_width
    min_y += reduce_height
    max_x -= reduce_width
    max_y -= reduce_height


def check_detect_area(frame):
    cv2.line(frame, (min_x, min_y), (min_x, max_y), WHITE_COLOR, 2)
    cv2.line(frame, (min_x, max_y), (max_x, max_y), WHITE_COLOR, 2)
    cv2.line(frame, (min_x, min_y), (max_x, min_y), WHITE_COLOR, 2)
    cv2.line(frame, (max_x, min_y), (max_x, max_y), WHITE_COLOR, 2)


def draw_landmark(frame, rect):
    if rect is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (i, (x, y)) in enumerate(shape):
            cv2.circle(frame, (x, y), 2, color, -1)

def checkFaceCoordinate(face_coordinates, in_area=True):
    if len(face_coordinates) > 0:
        if in_area:
            for face in face_coordinates:
                (x, y, w, h) = face_utils.rect_to_bb(face)
                if x in range(min_x, max_x) and y in range(min_y, max_y) \
                        and x + w in range(min_x, max_x) and y + h in range(min_y, max_y):
                    return face, (x, y, w, h)
        else:
            face = face_coordinates[0]
            (x, y, w, h) = face_utils.rect_to_bb(face)
            return face, (x, y, w, h)
    return None, None

#sys.path.append("../")

windowName = 'Crowd Walk'
FACE_SHAPE = (255, 255)

isContinue = True
isArea = True
isLandmark = True
camera_width = 0
camera_height = 0
input_img = None
rect = None
bounding_box = None
mode_capture = False
img_counter = 1


##### 

def getCameraStreaming():
    capture = cv2.VideoCapture(0)
    global camera_width, camera_height
    camera_width = capture.get(3)
    camera_height = capture.get(4)
    set_default_min_max_area(camera_width, camera_height)
    if not capture:
        print("Failed to capture video streaming")
        sys.exit()
    print("Successed to capture video streaming")
    return capture

def setDefaultCameraSetting():
    cv2.startWindowThread()
    cv2.namedWindow(winname=windowName)
    cv2.setWindowProperty(winname=windowName, prop_id=cv2.WINDOW_FULLSCREEN, prop_value=cv2.WINDOW_FULLSCREEN)

def showScreenAndDetectFace(capture, color_ch=1):  #jj_add / for different emotion class models
    global isContinue, isGragh, isArea, isLandmark, input_img, rect, bounding_box, result, mode_capture, img_counter

    labels = ['Angry','Happy','Neutral']
    color_list = [RED_COLOR, dahong, WHITE_COLOR]
    n_label = len(labels)
    n_pic = 10
    
    while True:
        input_img, rect, bounding_box = None, None, None
        ret, frame = capture.read()
        face_coordinates = dlib_face_coordinates(frame)
        detect_area_driver(frame, face_coordinates,color_ch)
        
        cv2.putText(frame, "Press Q to quit",
                            (int(camera_width * 0.7), int(camera_height * 0.05)),
                            cur_font, 0.7, green, 1)
        
        if  mode_capture:
            if img_counter in range(1, n_pic*n_label+1):
                cv2.putText(frame, "Press space bar to capture",
                            (int(camera_width * 0.1), int(camera_height * 0.18)),
                            cur_font, 1.2, dahong, 2)
                cv2.putText(frame, "{} Img ({}/{})".format(labels[(img_counter-1)//n_pic],(img_counter-1)%n_pic+1, n_pic),
                        (int(camera_width * 0.35), int(camera_height * 0.85)),
                        cur_font, 0.8, color_list[(img_counter-1)//n_pic], 2)
         
            else:
                cv2.putText(frame, "Finish !", (int(camera_width * 0.31), int(camera_height * 0.15)),
                            cur_font, 0.7, cur_color, 2)
                mode_capture=False

            
        refreshScreen(frame)
        key = cv2.waitKey(20)
        
        if key == ord('s'): # 캡쳐 모드 시작!
            img_counter = 1
            #isContinue = not isContinue
            mode_capture = not mode_capture
        elif key == ord('l'): # land mark on off
            isLandmark = not isLandmark
        elif key == ord('o'):
            expend_detect_area()
        elif key == ord('p'):
            reduce_detect_area()
        elif key == ord('q'): # exit
            break
        elif key%256 == 32:  # jj_add / press space bar to save cropped gray image
            try:
                user_img_capture() #img_counter)
                #time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
                #img_name = './'+time_now+"_cropped_{}.png".format(img_counter)
                #cv2.imwrite(img_name, np.squeeze(input_img))#*255.))  # to recover normalized img to save as gray scale image
                #print("{} written!".format(img_name))
                #img_counter += 1
            except:
                print('Image can not be saved!')

def detect_area_driver(frame, face_coordinates, color_ch=1):
    global input_img, rect, bounding_box
    rect, bounding_box = checkFaceCoordinate(face_coordinates, isArea)

    # 얼굴을 detection 한 경우.
    if bounding_box is not None: #and isContinue:
     
        face = preprocess(frame, bounding_box, FACE_SHAPE)
        input_img = face
        #if face is not None:
            #input_img = np.expand_dims(face, axis=0)
            #input_img = np.stack((input_img,)*color_ch, -1 )
        if not mode_capture:
            cv2.putText(frame, "Press S to start capture", (int(camera_width * 0.2), int(camera_height * 0.18)),
                            cur_font, 1.1, dahong, 2)            
    else:
        cv2.putText(frame, "Please look at the camera :)", (int(camera_width * 0.1), int(camera_height * 0.5)),
                            cur_font, 1.1, green, 2)
            
            
def refreshScreen(frame):
    if isArea:
        check_detect_area(frame)
    if isLandmark:
        draw_landmark(frame, rect)
    drawFace(frame, bounding_box)
    cv2.imshow(windowName, frame)


def user_img_capture():
    global input_img, mode_capture, img_counter
    print(img_counter)
    try:
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
       
        if mode_capture:
            labels = ['Angry','Happy','Neutral']
        
            n_label = len(labels)
            n_pic = 10
        
            list_path=[0]*n_label
        
            for i,i_label in enumerate(labels):
                new_path = os.path.join(os.getcwd(),'data', i_label)
                list_path[i] = new_path
            
                if not os.path.exists(new_path):
                    os.makedirs(new_path)        
            
            img_name = os.path.join(list_path[(img_counter-1) // n_pic], time_now + "_pic_for_cw_{}.png".format(img_counter))
        else:
            img_name = os.path.joing(os.getcwd(),time_now + '_sample_pic_{}.png'.format(img_counter))
        
        cv2.imwrite(img_name, np.squeeze(input_img),params=[cv2.IMWRITE_PNG_COMPRESSION, 0])# to recover normalized img to save as gray scale image
        print("{} written!".format(img_name))
        img_counter += 1
    except:
        print('얼굴이 안보여요 ')
        
    #return img_counter

def main():
    print("Start main() function.")
    
    color_ch =3  # default for gray

    capture = getCameraStreaming()
    setDefaultCameraSetting()
    showScreenAndDetectFace(capture, color_ch)  #jj_add / for different emotion class models


if __name__ == '__main__':
    main()

cv2.destroyAllWindows()
