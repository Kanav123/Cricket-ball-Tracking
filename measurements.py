import cv2
import os
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

seq_dir = '/home/k2vats/Cricket_ball_tracking/bowling_sequences/seq2/'
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
sensitivity = 80
def my_key(name):
    return int(name.split('frame')[1].split('.')[0])

seq_names = sorted(os.listdir(seq_dir), key=my_key)
#print(seq_names)
#def remove_outlier():


def get_measurements():
    non_zero,meas = [], []
    for idx,img in enumerate(seq_names):
        path = seq_dir + img
        vidcap  = cv2.VideoCapture(path)
        success,frame = vidcap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red_0 = np.array([0, 130, 150]) 
        upper_red_0 = np.array([sensitivity, 255, 255])
        lower_red_1 = np.array([180 - sensitivity, 130, 150]) 
        upper_red_1 = np.array([180, 255, 255])

        mask_0 = cv2.inRange(hsv, lower_red_0 , upper_red_0);
        mask_1 = cv2.inRange(hsv, lower_red_1 , upper_red_1);
        
        mask = cv2.bitwise_or(mask_1, mask_0)
        mask = fgbg.apply(mask)
        non_zero.append(np.transpose(np.nonzero(mask)))

        #uncomment to show image and corresponding ball measurement position
        # cv2.imshow('Image',frame)
        # k = cv2.waitKey(0) & 0xff
        # if k == 27:
        #     break
        # cv2.imshow('Image',mask)
        # k = cv2.waitKey(0) & 0xff
        # if k == 27:
        #     break


    y_cord, x_cord = [], []
    #use mean of detected points as measurements
    for i, m in enumerate(non_zero):
        if not np.isnan(np.mean(m, axis=0)[0]):
            meas.append(np.mean(m, axis=0))
            y_cord.append(np.mean(m, axis=0)[0])
            x_cord.append(np.mean(m, axis=0)[1])

    return x_cord,y_cord




