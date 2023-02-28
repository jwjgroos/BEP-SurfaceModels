# -*- coding: utf-8 -*-
"""
Created on Sat May 22 21:51:58 2021

@author: jgroo
"""

import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as mat_io
#import subroutines_jmi as sj
import matplotlib.colors as clr
plt.close('all')
#temp = mat_io.loadmat('./data/cgray.mat')
#cgray = temp['cgray']
#temp = mat_io.loadmat('./data/cmap.mat')
#cmap = temp['cmap']
#temp = None

n_tot       = 1
n_test      = 0
n_train     = n_tot - n_test

i = 0
count = 0

# Specify the shot locations
combine = np.array([16,40,64,88,112])

# Initialize the arrays
x_train = np.zeros((n_train,64,128))
y_train = np.zeros((n_train,64,128))
x_test  = np.zeros((n_test,64,128))
y_test  = np.zeros((n_test,64,128))

# Combining all input channels and output channels
for xloc in combine:
    i = 0
    for train in range(1,n_train+1):
        train_append = np.load('./results/jmi_image/jmi_image_'+str(train)+'_'+str(xloc)+'.npy')
        x_train[train-1] = train_append[1:,:]
        
    for test in range(n_train+1,n_train+n_test+1):
        train_append = np.load('./results/jmi_image/jmi_image_'+str(test)+'_'+str(xloc)+'.npy')
        x_test[i] = train_append[1:,:]
        i += 1
    np.save('./U-Net/Prefocusing_Approach/x_train'+str(xloc)   , x_train)
    np.save('./U-Net/Prefocusing_Approach/x_test'+str(xloc)    , x_test)

for train in range(1,n_train+1):
    train_append = np.load('./results/jmi_image/jmi_image_'+str(train)+'.npy')
    y_train[train-1] = train_append[1:,:]

i=0
for test in range(n_train+1,n_train+n_test+1):
    train_append = np.load('./results/jmi_image/jmi_image_'+str(test)+'.npy')
    y_test[i] = train_append[1:,:]
    i += 1
    np.save('./U-Net/Prefocusing_Approach/y_train_jmi'   , y_train)
    np.save('./U-Net/Prefocusing_Approach/y_test_jmi'    , y_test)

