# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 22:19:17 2018

@author: 2014_Joon_IBS
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:15:05 2018

@author: 2014_Joon_IBS
"""

import os
import cv2
import pandas as pd
import numpy as np
import glob
#
#from keras.models import model_from_json
#from keras.models import load_model
#from keras.utils import np_utils
#from keras.utils.vis_utils import plot_model

from sklearn.model_selection import train_test_split

import autokeras as ak

import shutil

from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

from sklearn.utils import class_weight
from keras import backend as K
# 8 x8 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))      
#class_label = ['background', 'body','nose', 'tail']
#class_label = ['happy', 'fear', 'funny', 'boring', 'dunno', 'relax']
class_label = ['1', '2', '3', '4', '5', '6']
split_label = ['train', 'validation', 'test']
n_class = len(class_label)
#path = '/data/'
data_path = r'F:\Data\gen\dlib'

pic_size = 96

### make train/val/test folder

def preprocess_keras(img):
    #img = crop_center(img)
    
    
    #print(img.shape)
    #img = np.array(img)/255.  # normalize
    #img = saturate_img(img, max_thr=10)
    
    
    img=np.array(img, np.uint8)
    img = clahe.apply(img)  # histogram equalizatio
    img=np.reshape(img,(pic_size,pic_size,1))
    return img

def aug_k_from_dir(path_dat, batch_size=32, total_size=100):
    z_r = 0.2
    datagen = ImageDataGenerator(zoom_range = 0.1, brightness_range = [0.5, 1.5], rotation_range = 20, width_shift_range=0.1, height_shift_range = 0.1, vertical_flip=False, horizontal_flip=True, preprocessing_function=preprocess_keras)
# fit parameters from data
    dum = []
    y_dum=[]
    count=0
    #path_t = r'F:\Data\crowd_crop'
    #'/Data/aug/mice/'
    path_aug = os.path.join(path_dat,'aug')
    test_gen = datagen.flow_from_directory(directory=path_dat, color_mode='grayscale', target_size =(pic_size,pic_size), class_mode='sparse', batch_size=batch_size) # categorical
    aa = test_gen.classes # class index

    class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(aa), # 순서 주의. 확인.
                                                     aa)
    norm_class_w = class_weights / np.sum(class_weights)
    norm_class_w = norm_class_w * total_size
    
    print(class_weights)
    class_size = int(total_size / n_class)
    print(class_size)
    ### path 없으면 생성
    if not os.path.exists(path_aug): # split dir 없으면 생성
        os.makedirs(path_aug)
    for i in class_label:
        path_aug_c = os.path.join(path_aug,i)
        if not os.path.exists(path_aug_c): # split dir 없으면 생성
            os.makedirs(path_aug_c)
    ####    
    for x_b, y_b in datagen.flow_from_directory(directory=path_dat, color_mode='grayscale', target_size =(pic_size,pic_size), class_mode='categorical', batch_size=batch_size): #, shuffle=False):
        #print(np.sum(x_b))
        batch_count=0
        flag_class=[True]*n_class # true 일때만 저장. class 마다 샘플 수 다를 경우, 특정 숫자 도달시 각자 False 로 변하도록.
        
        for i, i_sample in enumerate(x_b):
            #batch_count+=1
            xx = x_b[i]/255.
            dum.append(xx)
            y_dum.append(y_b[i])
        
            img_name = "_aug_{}.jpg".format(count)
            class_id = np.argmax(y_b[i]) # 현재 class id
            path_aug_c = os.path.join(path_aug,class_label[class_id])
            
            n_cur_class_file = len(glob.glob(path_aug_c+'/*.jpg'))
            
            if n_cur_class_file >=class_size: # 이제 해당 클래스는 충분히 파일 많으므로 False flag 꽂아줌
                flag_class[class_id]=False
            
            if flag_class[class_id]: #해당 class 에 아직 더 sample 이 추가되어야 하면
                path_name=os.path.join(path_aug_c,img_name)
                cv2.imwrite(path_name, x_b[i])        
            count+=1
            
        if not any(flag_class): # 모든 class 가 가득 차면
            print(count)
            break
        
    return np.array(dum), np.array(y_dum) # 필요한 경우 array 형태로 저장
    
def aug_k(x,y):
    datagen = ImageDataGenerator(vertical_flip=True, horizontla_flip=True, shuffle=False)#,rotation_range\)
# fit parameters from data
    datagen.fit(x)
    batch_size = 100
    dum = []
    for x_b, y_b in datagen.flow(x, y, batch_size=batch_size):
        # create a grid of 3x3 images
        for i in range(0, batch_size):
            #pyplot.subplot(330 + 1 + i)
            #pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
            #print(x[i].shape)
            dum.append(x_b[i])
        # show the plot
        #pyplot.show()
        break
    return np.array(dum)

def make_img_split(data_path):
    
    os.chdir(data_path)
    
    for k, k_class in enumerate(class_label):
        print('Class:{}\n'.format(k_class))
        final_path = os.path.join(data_path, k_class)
        files = glob.glob(final_path+'/*.jpg')
        
        n_files = len(files)
       # idx = np.arange(n_files)
        np.random.shuffle(files)
        
        f_train = files[0 : int(n_files*0.6)]
        f_val = files[int(n_files*0.6):int(n_files*0.8)]
        f_test = files[int(n_files*0.8):n_files]
        
        list_data=[f_train,f_val,f_test]
        
        for i, i_split in enumerate(list_data):
            print('split:{}'.format(split_label[i]))
            split_path = os.path.join(data_path, split_label[i], k_class)
            if not os.path.exists(split_path): # split dir 없으면 생성
                os.makedirs(split_path)
            for i_data in i_split:
                shutil.copy(i_data, split_path)
        
## dir put together
def put_dir(now_path):  # to put all different folders (ex- test, train, val, ...) into just distinct classes
    count = 0 
    count2 = 0
    for (path, dir, files) in os.walk(now_path):
        print('\n{}'.format(count))
        count += 1
        
        emotion_bool = [i in path for i in class_label] ## temp[i] 에 i 가 속해있기만 하면 true 
        #emotion_bool.index(true)
        
        if any(emotion_bool): # 어떤 emotion 폴더 감지되면
            idx_emotion = emotion_bool.index(True)
            new_path = os.path.join(now_path,class_label[idx_emotion])
            if not os.path.exists(new_path):  # 폴더 없으면 생성
                os.makedirs(new_path)
            print(len(files))
            for file in files:
                new_file = os.path.join(new_path,file)
                src_file = os.path.join(path,file)
                if os.path.isfile(new_file):
                    count2 +=1
                    print('\n')
                    print(count2)
                    new_file = new_path + '/new_'+ str(count2)+file
                    print(new_file)
                shutil.copy2(src_file, new_file)
            
        
def balance_dist(now_path = './', ratio = 0.35):
    #files = glob.glob(final_path+'/*.png')
    
    for (path, dir, files) in os.walk(now_path):
        n_files = len(files)
        n_final = int(n_files*ratio)   # file 을 이만큼 남기고 지울것
        
        #n_final = 4000
        np.random.shuffle(files)
        for filename in files:
            
            ext = os.path.splitext(filename)[-1]
            if ext == '.png':
                #print("%s/%s" % (path, filename))
                fullname = os.path.join(path, filename)
                os.remove(fullname)
                
            n_current = len(glob.glob(path+'/*.png'))
            if n_current <= n_final:  # 원한만큼 지우면 그만.
                print('\n dist:{}\n'.format(n_current))
                break
def crop_center(img, target_size=30):
    center_len = int(img.shape[0]/2)

    start = int(center_len-target_size/2-1)
    end = int(center_len+target_size/2-1)
#    print(center_len)    
 #   print('start{} end{}'.format(start,end))
    new_img = img[start:end, start:end]
    return new_img

def rot_img(img, max_angle=180):
    angle = np.random.randint(0,max_angle)
    rows, cols = img.shape[:2]
    rot = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    rot = cv2.warpAffine(img, rot,(cols,rows))
    return rot

def del_aug(now_path, ribbon='aug'):
    count = 0
    for (path, dir, files) in os.walk(now_path):
        for file in files:
            if ribbon in file:
                count+=1
                file_path = os.path.join(path,file)
                os.remove(file_path)
                
    print('{} aug data deleted.'.format(count))
                
        
       
def flip_img(img):
    idx = np.random.randint(0,3)
    if idx ==2:
        return img
    else:
        img = cv2.flip(img,idx)
        return img
    
def saturate_img(img, max_thr=100):
    idx = np.random.randint(0,2)
    if idx ==1:
        thr = np.random.uniform(0,max_thr)
        img= cv2.add(img, thr)
        return img
    else:
        return img

def preprocess_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # load img as grayscale
    img = clahe.apply(img)  # histogram equalization
    img = np.array(img)/255.  # normalize
    return img


        
        

if __name__ == "__main__":
    print('Something start')
    #data_path = r'F:\Data\crowd_crop'
    #data_path = r'F:\Data\gen\dlib'
    #### my own aug 
    #load_img_gen_aug(data_path, save_img=True, n_aug=2000)
    
    ##### augmentation using keras
    x, y  = aug_k_from_dir(data_path, batch_size = 32, total_size = 6000)
    np.save('x.npy', x)
    np.save('y.npy', y)
    
    
    
#    
    
    