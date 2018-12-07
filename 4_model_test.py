# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:23:00 2018

@author: 2014_Joon_IBS
"""

from sklearn.model_selection import train_test_split
from keras.models import load_model
import cv2
import os
import glob 
import numpy as np
import pandas as pd
import csv
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))   

# 폴더/ 파일 이름을 숫자 순서대로 접근해서, 각 data 를 model 에 넣고, 그 prediction 값을 순차적으로 저장
#now_path = '/Data/fer_ck_cam_3_img/webcam_data/'
def pred_from_img_path(model, now_path):
    #now_path = '/Data/fer_ck_cam_3_img/fer_3class/'
    
    os.chdir(now_path)
    list_result = [] 
    list_files = []
    list_data = []
    
    list_dir = glob.glob(now_path+'/*/') # listdir, walk 썼으나, 아래 sort 시 문제로 변경.
    list_dir.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) # 숫자 순으로 정렬. 기본은 string 이라 제대로 정렬되지 않음.
    
    for i, i_dir in enumerate(list_dir):
        path_sub = os.path.join(now_path, i_dir)
        files = glob.glob(path_sub+'/*.jpg')
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) # 숫자 순으로 정렬. 기본은 string 이라 제대로 정렬되지 않음.
        print(i_dir+': '+str(len(files)))
        
        list_files.append(files)
        
        for filename in files:
            #file_path = os.path.join(path_sub, filename)
            img = preprocess_img(filename)
            result = model.predict(img)
            
            list_result.append(result)
#            list_data.append(img)
#    
#    list_data = np.array(list_data)
#    list_data = np.squeeze(list_data, axis=1)
#    list_data = np.squeeze(list_data, axis=3)
#    np.save('test_data.npy',list_data)
#        
    return list_result, list_dir, list_files

def preprocess_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  #load img as grayscale
    img = clahe.apply(img)  # histogram equalization
    img = np.array(img)/255.  # normalize
    img = img.reshape(-1, 40, 40,1) # to make input dimension same as model
    return img



# load img

# preprocess

# for list save
def csv_write(result, name): # 
    with open("output_{}.csv".format(name),'wb') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerows(result)

def pred_from_file(model, data_path):
    result, list_dir, list_files = pred_from_img_path(model, data_path)
    
    np_result = np.array(result) # convert into numpy array
    np_result = np.squeeze(np_result, axis=1)
    np.savetxt("mice_result.csv", np_result, delimiter=",")
    
    print(np.shape(np_result)) # result shape check
    
    list_dir = pd.DataFrame(list_dir)
    list_files = pd.DataFrame(list_files)
    
    list_dir.to_csv('dir_result.csv')
    list_files.to_csv('files_result.csv')
    
def main():
    #    # for test
#    path2 = '/Data/mice/nose'
#    lists = glob.glob(path2+'/*.jpg')
    ##
    #data_path = '/Data/image_patch_targeted'
    
    #model_path = '/Data/mice_transfer.h5'
    target_size=96
    data_path = r'F:\Data\gen\crowd_aug'
    os.chdir(data_path)
    model_path=r'F:\python\autokeras\ak_crowd_96.h5'
    ### data load
    print('\n#####################\nload npy\n')
    x_data = np.load('./x.npy')  # 7 means total 7 members, not 7 emotion classes :)
    y_data = np.load('./y.npy')    
    x_data = x_data.reshape(-1, target_size, target_size,1)
    #y_data = np.argmax(y_data, axis=1) # convert one hot encoding to catogorical integer
   
          
    ## train test split
   
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, shuffle = True, random_state=33)
    
    
    #### npy data. more speedy!
    
    #test_data = np.expand_dims(test_data,axis=3)
    #test_data = test_data[:,5:35,5:35,:]
    
    #### Prepare model
    model=load_model(model_path)
    model.compile(loss = categorical_crossentropy,
              optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7),
              metrics = ['accuracy'])    
    
    #### prediction using model.
    pred_result = model.predict(x_test) # predict for each class
    np.savetxt("result_npy.csv", pred_result, delimiter=",")
    
    
# save result
    
if __name__ == '__main__':
    print('start')
    main()
