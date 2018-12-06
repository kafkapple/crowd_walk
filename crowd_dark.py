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

#import tensorflow as tf
#import tensorflow.keras as keras
#from tensorflow.keras.models import load_model
import keras
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

import matplotlib.patches as mpatches

plt.style.use('dark_background')
model_weight_path = r'F:\github\fer\model\models\ak8.h5'


model_weight_path = './src/ak8.h5' #r'F:\github\fer\model\models\ak8.h5'

emotion = ['Angry', 'Happy', 'Neutral']
model = load_model(model_weight_path)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
angry_check = 0
# dlib을 위한 변수
landmarks = './src/shape_predictor_68_face_landmarks.dat' #os.path.join(os.getcwd(),'src','shape_predictor_68_face_landmarks.dat')
#landmarks = './src/shape_predictor_68_face_landmarks.dat'  # jj_modify for relative path to the dat
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmarks)

FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"

from colormap import hex2rgb


RED_COLOR = (0, 0, 255)
WHITE_COLOR = (255, 255, 255)
pink = (255,139,148)
green = (255, 170, 165) # 하늘색? 에메랄드 그린
dahong = (254,74,173)#(241, 153, 160)
pastel_rainbow = ['#a8e6cf','#dcedc1','#ffd3b6','#ffaaa5', '#ff8b94']
fig_width=400
#plot_fig = np.random.random((480,fig_width*2,3))

cur_color = pink#dahong#RED_COLOR#(255, 170, 165) # 에메랄드 그린
color = hex2rgb(pastel_rainbow[0])

#color = dahong #pink

cur_font = cv2.FONT_HERSHEY_DUPLEX

cam_width, cam_height = 0, 0
expand_width, expand_height = 0, 0
reduce_width, reduce_height = 0, 0
min_x, max_x, min_y, max_y = 0, 0, 0, 0

#(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
#(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

emotion_hist = []
labels = ['Angry','Happy','Neutral']

#sys.path.append("../")

windowName = 'Crowd Walk'
FACE_SHAPE = (48, 48)

isContinue = True
isArea = True
isLandmark = True
isGraph = True
flag_curr = True

camera_width = 0
camera_height = 0
input_img = None
rect = None
bounding_box = None
mode_capture = False
img_counter = 1
face_68=np.zeros((68,2))

plt.rcParams['figure.constrained_layout.use'] = True
## Main
fig_size= (10,6)

fig_total, axes_list = plt.subplots(1,2)

axes_list[0].set(ylabel='%', xlabel='Time')
#ax3.legend(loc='upper right')

label_test = ['A', 'H', 'N']
line1 = []
line2 = []
line3 = []
list_line = [line1, line2, line3]
    
for i in range(3):
        list_line[i], = axes_list[0].plot([], [], 'o-', linewidth=1)# label=labels[i])#,  color=pastel_rainbow[i])


def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
    buf = np.array(fig.canvas.renderer._renderer)
    return buf



def legend_patch(current_palette, labels):
    patches = []
    for i, _ in enumerate(labels):
        patch_i = mpatches.Patch(color=current_palette[i], label=labels[i])
        patches.append(patch_i)
    return patches


patches = legend_patch(pastel_rainbow, label_test) 
axes_list[1].legend(handles = patches, loc='best', edgecolor =None, fontsize=13, bbox_to_anchor=(0.8,0.7)) # upper right



for i_ax in axes_list:
    i_ax.spines['top'].set_visible(False)
    i_ax.spines['right'].set_visible(False)
    i_ax.grid(False)
    #i_ax.get_xticklines().set_visible(False)
    #i_ax.get_yticklines().set_visible(False)
    plt.setp(i_ax.get_xticklabels(), visible=False)
    plt.setp(i_ax.get_yticklabels(), visible=False)
    i_ax.tick_params(axis='both', which='both', length=10)



axes_list[0].set_xlim(auto=True)
axes_list[0].set_ylim((-5,120))

## bar graph
axes_list[1].set_ylim((0,100))



#im1 = ax1.imshow(dum)
#im2 = ax2.imshow(dum)
#im_plot = ax3.imshow(dum)

## Face


def dlib_face_coordinates(img):
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
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
        face_gray = clahe.apply(face_gray)/ 255.  # histogram equalization & normalize (0~1)
   
        return face_gray
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
    global face_68
    if rect is not None:
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray =frame
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        for (i, (x, y)) in enumerate(shape):
            cv2.circle(frame, (x, y), 2, color, -1)
        face_68 = shape # 68 x 2

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

##### 


def detect_area_driver(frame, face_coordinates, color_ch=1):
    global input_img, rect, bounding_box
    rect, bounding_box = checkFaceCoordinate(face_coordinates, isArea)

    # 얼굴을 detection 한 경우.
    if bounding_box is not None: #and isContinue:
     
        face = preprocess(frame, bounding_box, FACE_SHAPE)
        input_img = face
        input_img = np.squeeze(input_img)
        if face is not None:
            input_img = np.expand_dims(face, axis=0)
            input_img = np.stack((input_img,)*color_ch, -1 )
            

#        if not mode_capture:
#            cv2.putText(frame, "Press S to start capture", (int(camera_width * 0.2), int(camera_height * 0.18)),
#                            cur_font, 1.1, dahong, 2)            
#    else:
#        cv2.putText(frame, "Please look at the camera :)", (int(camera_width * 0.1), int(camera_height * 0.5)),
#                            cur_font, 1.1, green, 2)
#            
                

            
def refreshScreen(frame):
    global face_68
#    if isArea:
#        check_detect_area(frame) # 하얀 선으로 detect area 표시
        
    if isLandmark:
        draw_landmark(frame, rect)
    #drawFace(frame, bounding_box)
    cv2.imshow(windowName, frame) 
    #print(np.shape(np.array(frame)))
    
## Save    
def user_img_capture():
    global input_img, mode_capture, img_counter
    print(img_counter)
    try:
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
       
        if mode_capture:
            
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
#
#import pillow as PIL
#
#def showarray(a, fmt='jpeg'):
#    f = StringIO()
#    PIL.Image.fromarray(a).save(f, fmt)
    
#IPython.display.display(IPython.display.Image(data=f.getvalue()))

def change_width(ax, new_value):
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value
        # change bar width
        patch.set_width(new_value)
        # recenter
        patch.set_x(patch.get_x() + diff * .5)

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
    global isContinue, isGraph, isArea, isLandmark, input_img, rect, bounding_box, result, mode_capture, img_counter, face_68, emotion, model, flag_curr, emotion_hist, plot_fig
    start_t = time.time()
    labels = ['Angry','Happy','Neutral']
    color_list = [RED_COLOR, dahong, WHITE_COLOR]
    n_label = len(labels)
    n_pic = 10
    
    ### plot
    
  
    #### For live plot
    # plt.show()
    
    
    #fig = plt.figure()
    
    # axes = plt.gca()
    
    #for i in range(3):
    #list_plot[0], = axes_list[0].plot([], [], 'o-', linewidth=3, label=labels[0])
    n_bins=2
    ax_bar = axes_list[1].bar(range(n_bins), np.ones(n_bins)*10, color=pastel_rainbow[0:4:2])#,edgecolor='black')
    change_width(ax_bar, 0.7)
    #ax3.bar(np.arange(len(data)),data,)
    val_1 = 0
    val_2 = 0
    print('start')
    while True:
        cur_t = time.time()
        #result = [cur_t-start_t]*3  # test.
        
        input_img, rect, bounding_box = None, None, None
        ret, frame = capture.read()
        face_coordinates = dlib_face_coordinates(frame) #################################3 dlib coordinate
        ### face -> matplotlib        
        detect_area_driver(frame, face_coordinates,color_ch)        
        #frame_plt = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                
        plt.ion()               
        #result = np.mean(face_68) # 68x2
        #qprint(result)
#        cv2.putText(frame, "Press Q to quit",
#                            (int(camera_width * 0.7), int(camera_height * 0.05)),
#                            cur_font, 0.7, green, 1)
            
        if input_img is not None:
            result = model.predict(input_img)[0]
            #print(result)
            #print(np.mean(face_coordinates))

            # History saving
            emotion_hist.append(result)  # to track emotion history

            ## live plot
            n_emotion = len(emotion_hist)
            n_bin = 10
            n_data = 20
            # print(str(n_emotion)+'\n')

            if n_emotion % 1 == 0:
                if n_data < n_emotion and flag_curr:
                    #print('cutting')
                    emotion_hist_cur = emotion_hist[-1-n_data:]
                    n_emotion = len(emotion_hist_cur)
                    #print(n_emotion, emotion_hist_cur[0])
                else:
                    emotion_hist_cur = emotion_hist
                    n_emotion = len(emotion_hist_cur)
                    
                cur_bin = min(n_bin, n_emotion) 
                emotion_data = np.array(emotion_hist_cur)
            
                xdata = np.array(range(n_emotion))

                # 3종류 감정 각각 누적 데이터 plot
                #print('emotion:',np.mean(emotion_data, axis=0))
                
                    
                for i in range(3):
                    list_line[i].set_data(xdata, emotion_data[:,i] * 100) ## temp
                    
                axes_list[0].set_xlim((0, n_emotion))
                #axes_list[1].set_xlim((0, 3))
                #print(n_emotion)

                cum_1 = emotion_data[-cur_bin:,1]
                cum_2 = emotion_data[-cur_bin:,2]
                print('cum1',cum_1)
                val_1+=np.mean(cum_1, axis=0)*0.1
                val_2+=np.mean(cum_2, axis=0)*0.1
                print(val_1,val_2)
                #print(val_1, val_2)
                #mean_val  = np.mean(emotion_data)
                #emotion_hist[i]
                #list_line[1].set_data([1,2], [mean_val, mean_val**2] ) ## temp
                for bar_i, h in zip(ax_bar, [val_1,val_2]):
                    #print('h: ',h)
                    bar_i.set_height(h)
                

                #plt.draw()
                #plt.pause(0.1)
                
                
                
                plot_fig = fig2data(fig_total)
                plot_fig = cv2.resize(plot_fig, (fig_width*2,480))
                
                plot_fig = cv2.cvtColor(plot_fig, cv2.COLOR_RGB2BGR)
                #print('fig:',np.shape(plot_fig))
                
                
                
                # time.sleep(0.1)

                # add this if you don't want l the window to disappear at the end
                plt.show(fig_total)
      
        
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
                        
        #plt.pause(0.1)
        #plt.ioff()
        #plt.show()
        
        #print(np.shape(frame))
        frame_concat = np.concatenate((frame, plot_fig), axis=1)
        #frame = plot_fig
#        b, g, r = cv2.split(frame_concat)   # img파일을 b,g,r로 분리
#        frame_concat = cv2.merge([r,g,b]) # b, r을 바꿔서 Merge
        refreshScreen(frame_concat)
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
        elif key == ord('c'):
            flag_curr = not flag_curr
            print('plot mode change')
            
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

#
#import dash
#from dash.dependencies import Output, Event
#
#import dash_core_components as dcc
#import dash_html_components as html
#import plotly
#import random
#import plotly.graph_objs as go
#from collections import deque
#import time
#import pandas as pd
#
#
#X = deque(maxlen=20)
#Y = deque(maxlen=20)
#X.append(1)
#Y.append(1)
#
#app = dash.Dash(__name__)
#app.layout = html.Div(
#    [
#         dcc.Graph(id='live-graph', animate=True),
#         dcc.Interval(
#            id='graph-update',
#            interval=1000 # per 1 sec
#            ),
#         ]
#    )
#@app.callback(Output('live-graph','figure'),
#            events = [Event('graph-update', 'interval')]) 
#
#
#def update_graph():
#    global X
#    global Y
#    global emotion_hist
#    
#    df = pd.DataFrame(emotion_hist)
#    if len(df) > 10:
#        
#        df = df.rename(columns={0:'happy',1:'angry',2:'neutral'})
#        X = df.index
#        Y = df['happy']
#    else:
#        X.append(X[-1]+1)
#        Y.append(Y[-1]+Y[-1]*random.uniform(-0.1,0.1))
#    
### build data
#    data = go.Scatter(
#            y = list(Y),
#            x = list(X),
#            name='Nier', # 'Scatter'
#            mode='lines+markers'
#            )
#    return {'data':[data], 'layout': go.Layout(xaxis = dict(range=[min(X), max(X)]), yaxis = dict(range=[0, 100]))}

def main():
    print("Start main() function.")
    
    color_ch =1  # default for gray
    
    #showScreenAndDetectFace(capture, color_ch)  # jj_add / for different emotion class models

    capture = getCameraStreaming()
    setDefaultCameraSetting()
    showScreenAndDetectFace(capture, color_ch)  #jj_add / for different emotion class models


if __name__ == '__main__':
    plt.close('all')
    #app.run_server(debug=True)
    main()
    
    

cv2.destroyAllWindows()
