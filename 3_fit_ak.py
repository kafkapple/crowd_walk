# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 22:21:53 2018

@author: 2014_Joon_IBS
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 02:04:40 2018

@author: 2014_Joon_IBS
"""

import autokeras as ak
from autokeras.preprocessor import OneHotEncoder, DataTransformer
from autokeras.constant import Constant

from keras.models import model_from_json
from keras.models import load_model
from keras.utils import np_utils

from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix #classification_report

import itertools  # for confusion matrix plot
import cv2
import os
import numpy as np
from torchvision import models
import pandas as pd
#from autokeras.classifier import load_image_dataset
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt
import glob

def plot_hist(hist):
    plt.figure(0)
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='lower left')

    plt.show()
    fig.savefig('loss_accuracy_plot')
    

# Make and plot confusion matrix. To see the detailed imformation about TP, TN of each classes.    
def make_confusion_matrix(model, x, y, normalize = True):
    predicted = model.predict(x)

    pred_list = []; actual_list = []
    for i in predicted:
        pred_list.append(np.argmax(i))
    for i in y:
        actual_list.append(np.argmax(i))

    confusion_result = confusion_matrix(actual_list, pred_list)
    plot_confusion_matrix(confusion_result, classes = class_label, normalize = normalize, title = 'Confusion_matrix')
    return confusion_result

def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues):  
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("normalized")
    else:
        print('without normalization')

    print(cm)
    plt.figure(1)
    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Confusion_matrix')

def model_fit(x_train, y_train, resume = True, iter_num=10, time_limit = 12):
    print('Model training start')
    ##clf = ak.ImageClassifier(verbose = True, searcher_args={'trainer_args':{'max_iter_num':10}}, path = './', resume=resume) 
    clf = ak.image_classifier.ImageClassifier(verbose = True, searcher_args={'trainer_args':{'max_iter_num':iter_num}}, path = './', resume=resume) 

    print('fit start')
    out = clf.fit(x_train, y_train, time_limit = time_limit*60*60 )
    
    print('out: ', out)
    return clf
    
def final_fit(clf, x_train, y_train, x_test, y_test, iter_num =5):
    print('final fit')
    final_out = clf.final_fit(x_train, y_train, x_test, y_test, retrain=False, trainer_args={'max_iter_num':iter_num})
    print('final out: ', final_out)
 
    results = clf.predict(x_test)
    print('predict: ', results)

    y = clf.evaluate(x_test, y_test)
    print('eval: ', y)
    return clf

def keras_train(model, x_train, y_train, epoch):
    
    model.compile(loss = categorical_crossentropy,
              optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7),
              metrics = ['accuracy'])
    early_stopping = EarlyStopping(monitor = 'val_acc', min_delta = 0.01, patience = 10, 
                                       verbose = 1, mode = 'max')
    
    # fit (learning) start. 1-epoch means entire data set.
    # during training, monitor the validation accuracy using validation data (splitted from training data ex -  0.2).
    hist = model.fit(x_train, y_train, 
                          validation_split = 0.2, 
                          shuffle = True, 
                          batch_size = 32, epochs = epoch, verbose = 1, 
                          callbacks = [early_stopping] )
    
    return model, hist

def keras_test(model, hist, x_test, y_test):
    scores = model.evaluate(x_test, y_test, batch_size = 32)
    print("Loss:{}\nAccuracy:{}".format(scores[0],scores[1]))
    plot_hist(hist)            
    confusion_result = make_confusion_matrix(model, x_test, y_test, normalize = True) 

def show_best_model(clf):
    best_model_id = clf.get_best_model_id()
    print('\nBest model id: {}\n '.format(best_model_id))
    best_model = clf.load_searcher().load_best_model()
    print('n_layer: ', best_model.n_layers)
    print(best_model.produce_model())
    
    keras_model = best_model.produce_keras_model()
    #save_model_weight(keras_model)
    
    #keras_model.save('./' + model_name + '.h5')
    print('\nSave best model.\n')
    
    plot_model(keras_model, to_file = 'best_net_id_{}.png'.format(best_model_id), show_shapes=True, show_layer_names=True)
    
    return keras_model

# from current path
def load_ak_show_best_id(path = './'):
    clf = ak.image_classifier.ImageClassifier(verbose = True, 
                                          searcher_args={'trainer_args':{'max_iter_num':1}}, path = path, resume=True)     
    
    best_model_id = clf.get_best_model_id()
    print('Best model id: {} '.format(best_model_id))    
    best_model = clf.load_searcher().load_best_model()
    print('n_layer: ', best_model.n_layers)
    #print(best_model.produce_model())
    return best_model

if __name__ == '__main__':
    

    # 1. data load
    
    
    target_size=48
    class_label = ['1', '2', '3', '4', '5', '6']
    n_class = len(class_label)
    #data_path = r'F:\Data\gen\crowd_aug'
    #'/github/neuro/'
    #os.chdir(data_path)
    #list_files = os.listdir(data_path)
    
    ## npy data load
    x_data = np.load('./x_{}.npy'.format(target_size))  # 7 means total 7 members, not 7 emotion classes :)
    y_data = np.load('./y_{}.npy'.format(target_size))    
    x_data = x_data.reshape(-1, target_size, target_size,1)
    y_data = np.argmax(y_data, axis=1) # convert one hot encoding to catogorical integer
    print(np.shape(y_data))
    print('\n#####################\nload npy\n')
          
          
    os.chdir('/python/autokeras/')    
    
    
    class_dist = [len(y_data[y_data==i]) for i, c in enumerate(class_label)] 
    
    print('\nClass distribution:{0}\n'.format(class_dist))
    # train +val / test set split 
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, shuffle = True, random_state=33)
    
    ######################### Autokeras fit start.
    clf = model_fit(x_train, y_train, resume=True, iter_num=10, time_limit = 9)
    
    ##### load best model
    print('load best model')
    best_model = load_ak_show_best_id()
    
    keras_model = best_model.produce_keras_model()
    plot_model(keras_model, to_file = 'best_net_crowd.png', show_shapes=True, show_layer_names=True)
    
    #print('\nFinal fit start.\n')
    #clf = final_fit(best_model, x_train, y_train, x_test, y_test)
    model_name = 'ak_crowd'
    
    #### Final best model training. change y encoding
    #y_train = np_utils.to_categorical(y_train, n_class)  # to convert one-hot encoding
    #y_test = np_utils.to_categorical(y_test, n_class)
    
    #keras_model, hist = keras_train(keras_model, x_train, y_train, epoch=10)
    #keras_test(keras_model, hist, x_test,y_test)
    print('\nSave best model.\n')
    keras_model.save('./' + model_name + '.h5')
   # keras_model = save_cur_model(clf, model_name='ak_now')