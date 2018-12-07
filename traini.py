# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:14:01 2018

@author: 2014_Joon_IBS
"""

from keras.models import load_model
from keras.utils import np_utils
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import plot as pl

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))      
class_label = ['background', 'body','nose', 'tail']
n_class = len(class_label)
target_size = 30
epochs = 20
batch_size = 16
data_path ='/Data/mice/aug/split'    
model_path = '/python/autokeras/ak_white.h5'
    


def preprocess_img(img):    
    img=img.astype(np.uint8) # this is to match keras type? but uint8 might be more universal
    #img = clahe.apply(img) 
    img = img/255.  # normalize
    img = np.reshape(img, (target_size,target_size,1))    
    return img


early_stopping = EarlyStopping(monitor = 'val_acc', min_delta = 0.01, patience = 10, 
                               verbose = 1, mode = 'max')

## Fit generator with augmentation

def gen_train(model, data_path=data_path):     
    train_datagen = ImageDataGenerator(
               
                rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
                # randomly shift images horizontally (fraction of total width)
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                cval=0.,  # value used for fill_mode = "constant"
                horizontal_flip=True,  # randomly flip images
                vertical_flip=True,  # randomly flip images
                # set rescaling factor (applied before any other transformation)
                #rescale=1./255,
                # set function that will be applied on each input
                preprocessing_function=preprocess_img,
                # image data format, either "channels_first" or "channels_last"
                data_format=None, 
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0)
        
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_img)#rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
            data_path+'/train/', shuffle=True,
            target_size=(target_size, target_size),
            #batch_size=batch_size, 
            color_mode = 'grayscale',
            class_mode='categorical') #,  save_to_dir = data_path+'aug', save_prefix='aug_',) # option is for monitoring data augmentation
    
    val_generator = test_datagen.flow_from_directory(
            data_path+'/validation/', shuffle=True,
            target_size=(target_size, target_size),
            #batch_size=batch_size, 
            color_mode = 'grayscale',
            class_mode='categorical') #,  save_to_dir = data_path+'aug', save_prefix='aug_',) # option is for monitoring data augmentation
    
    test_generator = test_datagen.flow_from_directory(
            data_path+'/test/', shuffle=False,
            target_size=(target_size, target_size),
            #batch_size=batch_size, 
            color_mode = 'grayscale',
            class_mode='categorical') #,  save_to_dir = data_path+'aug', save_prefix='aug_',) # option is for monitoring data augmentation
    
    
    hist = model.fit_generator(
            train_generator,
            #steps_per_epoch=30,
            epochs=epochs, verbose=1,
            validation_data=val_generator,
            #validation_steps=10,
            callbacks = [early_stopping])
    
    return hist, test_generator

### evaluation result for generator / flow from dir
def gen_eval(model, test_gen):
    pred = model.predict_generator(test_gen)
    
    y_true = test_gen.classes
    y_pred = np.argmax(pred,axis=1)

    confusion_result = confusion_matrix(y_true, y_pred)
    pl.plot_confusion_matrix(confusion_result, classes = class_label, normalize = True, title = 'Confusion_matrix')

def main():
    
    model=load_model(model_path)
    model.summary()
    
    model.compile(loss = categorical_crossentropy,
              optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7),
              metrics = ['accuracy'])

    hist, test_gen = gen_train(model, data_path)
    
    #### result
    pl.plot_hist(hist)
    gen_eval(model, test_gen)
    
    model.save('new_white_2.h5')
    
    
if __name__ =='__main__':
    main()
 