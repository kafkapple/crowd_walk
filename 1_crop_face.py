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

#face_shape = (48, 48)
face_shape = (256, 192)

RED_COLOR = (0, 0, 255)
WHITE_COLOR = (255, 255, 255)
pink = (255,139,148)
green = (255, 170, 165) # 하늘색? 에메랄드 그린
dahong = (254,74,173)#(241, 153, 160)


   
        
def preprocess(img):
    #face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_gray = clahe.apply(img)#/ 255.  # histogram equalization & normalize (0~1)
    return face_gray
    
def dlib_face_coordinates(img):
    gray = preprocess(img)
    face_coordinates = detector(gray, 0)
    rect, face_coordinates = checkFaceCoordinate(face_coordinates, True)
    return gray, face_coordinates

def checkFaceCoordinate(face_coordinates, in_area=True):
    if len(face_coordinates) > 0:
        if in_area:
            for face in face_coordinates:
                (x, y, w, h) = face_utils.rect_to_bb(face)
                return face, (x,y,w,h)
                
        else:
            face = face_coordinates[0]
            (x, y, w, h) = face_utils.rect_to_bb(face)
            return face, (x, y, w, h)
    return None, None
#
        
##########
#    if not os.path.exists(path + 'dlib/'):
#        os.makedirs(path + 'dlib/')
#        
#    for c_i, class_i in enumerate(class_label):
#        if not os.path.exists(path + 'dlib/' + class_i): # if there's no class folder, make it
#            os.makedirs(path + 'dlib/' + class_i)    
#                
#    landmarks = []
#    for i, picture in enumerate(pictures): 
#        xlist = []
#        ylist = []
#        picture = picture.reshape((48, 48))
#        img_name = 'dlib_%d.jpg' % i
#        path_name = os.path.join(path, 'dlib/', class_label[labels[i]], img_name)
#        #path_name = path + class_label[labels[i]] + img_name
#        #cv2.imwrite(path_name, picture)
#       
# 
#
#        gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
#        
#        clahe_image = clahe.apply(gray)
#        detections = detector(clahe_image, 1) #Detect the faces in the image
#        for k,d in enumerate(detections): #For each detected face
#            shape = predictor(clahe_image, d) #Get coordinates
#            for i in range(1,68): # To draw 68 landmarks from dlib
#                cv2.circle(picture, (shape.part(i).x, shape.part(i).y), 1, (0,0,200), thickness=1) 
#                xlist.append(float(shape.part(i).x))
#                ylist.append(float(shape.part(i).y))
#                #For each point, draw a red circle with thickness2 on the original frame
#                
#            
#            for x, y in zip(xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.
#                landmarks.append(x)
#                landmarks.append(y)
#    
#        #cv2.imwrite(os.path.join(path,class_label[labels[i]], img_name), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
#        cv2.imwrite(path_name, picture, [cv2.IMWRITE_PNG_COMPRESSION, 0])
#            
#def convert_dlib(picture):
#    gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
#    clahe_image = clahe.apply(gray)
#    detections = detector(clahe_image, 1) #Detect the faces in the image
#    for k,d in enumerate(detections): #For each detected face
#        shape = predictor(clahe_image, d) #Get coordinates
#        for i in range(1,68): # To draw 68 landmarks from dlib
#            cv2.circle(picture, (shape.part(i).x, shape.part(i).y), 1, (0,0,200), thickness=1) 
#            xlist.append(float(shape.part(i).x))
#            ylist.append(float(shape.part(i).y))
#            #For each point, draw a red circle with thickness2 on the original frame
#            
#        for x, y in zip(xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.
#            landmarks.append(x)
#            landmarks.append(y)
#
#    #cv2.imwrite(os.path.join(path,class_label[labels[i]], img_name), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
#    face_coordinates = dlib_face_coordinates(picture) 
#    face = crop_face(picture, face_coordinates)
#    
#    if face is not None:
#        print('Found face')
#        face_resize = cv2.resize(face, face_shape)
#        face_gray = cv2.cvtColor(face_resize, cv2.COLOR_BGR2GRAY)
#        #face_gray = clahe.apply(face_gray)/ 255.  # histogram equalization & normalize (0~1)
#        
#    cv2.imwrite( face_gray, [cv2.IMWRITE_PNG_COMPRESSION, 0])
# 
def check_resize_area(face_coordinates):
    (x, y, w, h) = face_coordinates
    if x - int(w / 4) > 0 and y - int(h / 4) > 0:
        return True
    return False

def crop_face(frame, face_coordinates):
    cropped_img = frame
    (x, y, w, h) = face_coordinates
    #if check_resize_area(face_coordinates):
    #print('chk resize area')
    cropped_img = frame[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
    #cv2.imwrite('./face.jpg', cropped_img, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
    return cropped_img
    #else:
    #    return None
    
import glob
def main():
    print("Start main() function.")
    class_label = ['1', '2', '3', '4', '5', '6']
    color_ch =1  # default for gray
    #path = r'F:\Data\crowd_crop' # current location
    path = r'F:\Data\crowd_face'
    path_save = r'F:\Data\gen'
    #os.chdir(cur_loc) # current location
    #pic_list = glob.glob('*.jpg')   
    
    path_aug = os.path.join(path_save,'dlib_origin')
    if not os.path.exists(path_aug): # split dir 없으면 생성
        os.makedirs(path_aug)
    for i in class_label:
        path_aug_c = os.path.join(path_aug,i)
        if not os.path.exists(path_aug_c): # split dir 없으면 생성
            os.makedirs(path_aug_c)
    list_dir = os.listdir(path)
    
    # 폴더 내 탐색. jpg 
    #for x in os.walk(path):
    for x_dir in list_dir:
        cur_loc = os.path.join(path, x_dir)
        os.chdir(cur_loc)
        pic_list = []
        count=0
        count_class = 0
        for i_pic in glob.glob(os.path.join(cur_loc, '*.jpg')):
            pic_list.append(i_pic)
            
            #for i_pic in pic_list:
            #print(i_pic)
            #i_pic = cv2.imread(i_pic)
            #### img read 
            i_pic = cv2.imread(i_pic, cv2.IMREAD_GRAYSCALE)
            gray_face, face_coordinates = dlib_face_coordinates(i_pic) 
            
#            ## margin 
#            (x,y,w,h) = face_coordinates 
#            margin_crop = 0.1
#            #x= int(x -x*margin_crop)
#            y=int(y-y*margin_crop*3)
#            w=int(w+w*margin_crop)
#            h=int(h+h*margin_crop)
#            face_coordinates = (x,y,w,h)
            
            gray_face = crop_face(gray_face, face_coordinates)
            target_size = 48
            face_shape=(target_size,target_size)
            
            gray_face = cv2.resize(gray_face, face_shape)
            path_aug_final = os.path.join(path_aug, x_dir,str(count)+'.jpg')
            cv2.imwrite(path_aug_final, gray_face)
            count+=1
        count_class+=1
        
    #showScreenAndDetectFace(capture, color_ch)  # jj_add / for different emotion class models

if __name__ == '__main__':
    
    main()
    