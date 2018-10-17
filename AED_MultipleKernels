
from __future__ import print_function 
import sys
import cPickle
import numpy as np
import argparse
import glob
import time
import os

import keras
from keras import backend as K
from keras.models import Sequential,Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute,Lambda, RepeatVector
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Conv2D,MaxPooling2D, MaxPooling3D, Convolution1D,MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import Merge, Input, merge
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
import h5py
from keras.layers.merge import Multiply
from sklearn import preprocessing
import scipy.io as sio
import random
from keras.backend.tensorflow_backend import set_session
import config as cfg
import tensorflow as tf
from prepare_data import create_folder, load_hdf5_data, do_scale
from data_generator import RatioDataGenerator
from evaluation import io_task4, evaluate
from keras.regularizers import l2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
'''
GPU = "1"
# use specific GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))
'''
# CNN with Gated linear unit (GLU) block
def block(input):
    cnn = Conv2D(128, (3, 3), padding="same", activation="linear", use_bias=False)(input)
    cnn = BatchNormalization(axis=-1)(cnn)

    cnn1 = Lambda(slice1, output_shape=slice1_output_shape)(cnn)
    cnn2 = Lambda(slice2, output_shape=slice2_output_shape)(cnn)

    cnn1 = Activation('relu')(cnn1)
    cnn2 = Activation('softmax')(cnn2)

    out = Multiply()([cnn1, cnn2])
    return out
    
def slice1(x):
    return x[:, :, :, 0:64]

def slice2(x):
    return x[:, :, :, 64:128]

def slice3(x):
    return x[:, :, :, 128:192]    

def slice1_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

def slice2_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

def slice3_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])    

# Attention weighted sum
def outfunc(vects):
    cla, att = vects    # (N, n_time, n_out), (N, n_time, n_out)
    att = K.clip(att, 1e-7, 1.)
    out = K.sum(cla * att, axis=1) / K.sum(att, axis=1)     # (N, n_out)
    return out    

#def mil_squared_error(y_true, y_pred):
#    return K.square(K.max(y_pred) - K.max(y_true))
# Train model
def train(args):
    num_classes = cfg.num_classes
    

    args.tr_hdf5_path="/data/users/21799506/Data/DCASE2017_Task4/packed_features/logmel/training.h5" 
    args.te_hdf5_path="/data/users/21799506/Data/DCASE2017_Task4/packed_features/logmel/testing.h5" 
    args.scaler_path="/data/users/21799506/Data/DCASE2017_Task4/scalers/logmel/training.scaler" 
    args.out_model_dir="/data/users/21799506/Data/DCASE2017_Task4/models/crnn_sed_inception_3to6_gated_v2"
    # Load training & testing data
    (tr_x, tr_y, tr_na_list) = load_hdf5_data(args.tr_hdf5_path, verbose=1)
    (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    print("tr_x.shape: %s" % (tr_x.shape,))

    # Scale data
    tr_x = do_scale(tr_x, args.scaler_path, verbose=1)
    te_x = do_scale(te_x, args.scaler_path, verbose=1)
    
    # Build model
    (_, n_time, n_freq) = tr_x.shape    # (N, 240, 64)
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 240, 64)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 240, 64, 1)

    #conv1 = Conv2D(128,(3,3), padding="same", activation="relu", W_regularizer=l2(0.0002))(a1)
    #conv1 = MaxPooling2D(pool_size=(3,3))(conv1)
    #conv1 = Conv2D(64,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    #conv1 = MaxPooling2D(pool_size=(3,3))(conv1)
    #conv1 = Conv2D(192,(3,3), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    #conv1 = MaxPooling2D(pool_size=(1,2))(conv1)
     
    ##### First inception and the link to the output
    conv1 = a1
    #inception1 = Conv2D(64,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1) ### 
    #inception1 = MaxPooling2D(pool_size=(1,2))(inception1)
    #inception2 = Conv2D(96,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    #inception2 = Conv2D(64,(1,2),padding="same", activation="relu", W_regularizer=l2(0.0002))(inception2) ### 
    #inception2 = MaxPooling2D(pool_size=(1,2))(inception2)
    inception3 = Conv2D(64,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    inception3 = Conv2D(128,(1,3),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception3) ###
    inception3 = BatchNormalization(axis=-1)(inception3)
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception3)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception3)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception3 = Multiply()([cnn1, cnn2])
    inception3 = MaxPooling2D(pool_size=(1,2))(inception3)
    inception4 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    inception4 = Conv2D(128,(1,4),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception4) ### 
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception4)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception4)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception4 = Multiply()([cnn1, cnn2])    
    inception4 = MaxPooling2D(pool_size=(1,2))(inception4)
    inception5 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    inception5 = Conv2D(128,(1,5),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception5) ### 
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception5)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception5)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception5 = Multiply()([cnn1, cnn2])     
    inception5 = MaxPooling2D(pool_size=(1,2))(inception5)
    inception6 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    inception6 = Conv2D(128,(1,6),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception6) ###
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception6)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception6)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception6 = Multiply()([cnn1, cnn2]) 
    inception6 = MaxPooling2D(pool_size=(1,2))(inception6)
    #inception7 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    #inception7 = Conv2D(64,(1,7),padding="same", activation="relu", W_regularizer=l2(0.0002))(inception7) ### 
    #inception7 = MaxPooling2D(pool_size=(1,2))(inception7)
    #inception8 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    #inception8 = Conv2D(64,(1,8),padding="same", activation="relu", W_regularizer=l2(0.0002))(inception8) ### 
    #inception8 = MaxPooling2D(pool_size=(1,2))(inception8)            
    inception_max    = MaxPooling2D(pool_size=(1,2))(conv1)
    inception_max = Conv2D(128,(1,1),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception_max)
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception_max)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception_max)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception_max = Multiply()([cnn1, cnn2])     
    inception_out1 = merge([inception3,inception4,inception5,inception6,inception_max],mode="concat",concat_axis=3)   # N, 240, 32, 320    
    #loss_ave_pool1 = AveragePooling2D(pool_size=(5,5),strides=(3,3))(conv1)
    
    con = inception_out1
    loss1_conv = Conv2D(128,(1,4),padding='same',activation='relu',W_regularizer=l2(0.0002))(con)
    loss1_conv = MaxPooling2D(pool_size=(4,4))(loss1_conv)
    loss1_conv = Conv2D(128,(1,4),padding='same',activation='relu',W_regularizer=l2(0.0002))(con)
    loss1_conv = MaxPooling2D(pool_size=(3,2))(loss1_conv)
    loss1_conv = Conv2D(128,(1,4),padding='same',activation='relu',W_regularizer=l2(0.0002))(con)
    loss1_conv = MaxPooling2D(pool_size=(2,2))(loss1_conv) 
    loss1_conv = Conv2D(128,(1,4),padding='same',activation='relu',W_regularizer=l2(0.0002))(con)
    loss1_conv = MaxPooling2D(pool_size=(2,2))(loss1_conv)     
    loss1_flat = Flatten()(loss1_conv)   
    loss1_fc = Dense(64,activation='relu',W_regularizer=l2(0.0002))(loss1_flat)
    loss1_drop_fc = Dropout(0.7)(loss1_fc)
    loss1_classifier = Dense(17,W_regularizer=l2(0.0002))(loss1_drop_fc)
    loss_classifier_act1 = Activation('softmax')(loss1_classifier)
    
    ##### Second inception and the link to the output
    conv1 = inception_out1
    #inception1 = Conv2D(64,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1) ### 
    #inception1 = MaxPooling2D(pool_size=(1,2))(inception1)
    #inception2 = Conv2D(96,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    #inception2 = Conv2D(64,(1,2),padding="same", activation="relu", W_regularizer=l2(0.0002))(inception2) ### 
    #inception2 = MaxPooling2D(pool_size=(1,2))(inception2)
    inception3 = Conv2D(64,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    inception3 = Conv2D(128,(2,3),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception3) ###
    inception3 = BatchNormalization(axis=-1)(inception3)
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception3)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception3)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception3 = Multiply()([cnn1, cnn2])
    inception3 = MaxPooling2D(pool_size=(1,2))(inception3)
    inception4 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    inception4 = Conv2D(128,(2,4),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception4) ### 
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception4)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception4)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception4 = Multiply()([cnn1, cnn2])    
    inception4 = MaxPooling2D(pool_size=(1,2))(inception4)
    inception5 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    inception5 = Conv2D(128,(2,5),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception5) ### 
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception5)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception5)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception5 = Multiply()([cnn1, cnn2])     
    inception5 = MaxPooling2D(pool_size=(1,2))(inception5)
    inception6 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    inception6 = Conv2D(128,(2,6),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception6) ###
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception6)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception6)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception6 = Multiply()([cnn1, cnn2]) 
    inception6 = MaxPooling2D(pool_size=(1,2))(inception6)
    #inception7 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    #inception7 = Conv2D(64,(1,7),padding="same", activation="relu", W_regularizer=l2(0.0002))(inception7) ### 
    #inception7 = MaxPooling2D(pool_size=(1,2))(inception7)
    #inception8 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    #inception8 = Conv2D(64,(1,8),padding="same", activation="relu", W_regularizer=l2(0.0002))(inception8) ### 
    #inception8 = MaxPooling2D(pool_size=(1,2))(inception8)            
    inception_max    = MaxPooling2D(pool_size=(1,2))(conv1)
    inception_max = Conv2D(128,(1,1),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception_max)
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception_max)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception_max)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception_max = Multiply()([cnn1, cnn2])     
    inception_out2 = merge([inception3,inception4,inception5,inception6,inception_max],mode="concat",concat_axis=3)   # N, 240, 16, 320
    ## First link to the output
    
    con = inception_out2
    loss1_conv = Conv2D(128,(1,4),padding='same',activation='relu',W_regularizer=l2(0.0002))(con)
    loss1_conv = MaxPooling2D(pool_size=(4,4))(loss1_conv)
    loss1_conv = Conv2D(128,(1,4),padding='same',activation='relu',W_regularizer=l2(0.0002))(con)
    loss1_conv = MaxPooling2D(pool_size=(4,2))(loss1_conv)
    loss1_conv = Conv2D(128,(1,4),padding='same',activation='relu',W_regularizer=l2(0.0002))(con)
    loss1_conv = MaxPooling2D(pool_size=(3,2))(loss1_conv)    
    loss1_flat = Flatten()(loss1_conv)   
    loss1_fc = Dense(64,activation='relu',W_regularizer=l2(0.0002))(loss1_flat)
    loss1_drop_fc = Dropout(0.7)(loss1_fc)
    loss1_classifier = Dense(17,W_regularizer=l2(0.0002))(loss1_drop_fc)
    loss_classifier_act2 = Activation('softmax')(loss1_classifier) 
      
    ##### Third inception and the link to the output
    conv1 = inception_out2
    #inception1 = Conv2D(64,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1) ### 
    #inception1 = MaxPooling2D(pool_size=(1,2))(inception1)
    #inception2 = Conv2D(96,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    #inception2 = Conv2D(64,(1,2),padding="same", activation="relu", W_regularizer=l2(0.0002))(inception2) ### 
    #inception2 = MaxPooling2D(pool_size=(1,2))(inception2)
    inception3 = Conv2D(64,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    inception3 = Conv2D(128,(1,3),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception3) ###
    inception3 = BatchNormalization(axis=-1)(inception3)
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception3)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception3)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception3 = Multiply()([cnn1, cnn2])
    inception3 = MaxPooling2D(pool_size=(1,2))(inception3)
    inception4 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    inception4 = Conv2D(128,(1,4),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception4) ### 
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception4)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception4)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception4 = Multiply()([cnn1, cnn2])    
    inception4 = MaxPooling2D(pool_size=(1,2))(inception4)
    inception5 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    inception5 = Conv2D(128,(1,5),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception5) ### 
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception5)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception5)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception5 = Multiply()([cnn1, cnn2])     
    inception5 = MaxPooling2D(pool_size=(1,2))(inception5)
    inception6 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    inception6 = Conv2D(128,(1,6),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception6) ###
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception6)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception6)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception6 = Multiply()([cnn1, cnn2]) 
    inception6 = MaxPooling2D(pool_size=(1,2))(inception6)
    #inception7 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    #inception7 = Conv2D(64,(1,7),padding="same", activation="relu", W_regularizer=l2(0.0002))(inception7) ### 
    #inception7 = MaxPooling2D(pool_size=(1,2))(inception7)
    #inception8 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    #inception8 = Conv2D(64,(1,8),padding="same", activation="relu", W_regularizer=l2(0.0002))(inception8) ### 
    #inception8 = MaxPooling2D(pool_size=(1,2))(inception8)            
    inception_max    = MaxPooling2D(pool_size=(1,2))(conv1)
    inception_max = Conv2D(128,(1,1),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception_max)
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception_max)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception_max)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception_max = Multiply()([cnn1, cnn2])     
    inception_out3 = merge([inception3,inception4,inception5,inception6,inception_max],mode="concat",concat_axis=3)   # N, 240, 32, 256
    ## First link to the output
    #loss_ave_pool3 = AveragePooling2D(pool_size=(5,5),strides=(3,3))(inception_out3)
    #loss3_conv = Conv2D(128,(1,1),padding='same',activation='relu',W_regularizer=l2(0.0002))(loss_ave_pool3)  
    #loss3_flat = Flatten()(loss3_conv)
    #loss3_fc = Dense(1024,activation='relu',W_regularizer=l2(0.0002))(loss3_flat)
    #loss3_drop_fc = Dropout(0.7)(loss3_fc)
    #loss3_classifier = Dense(17,W_regularizer=l2(0.0002))(loss3_drop_fc)
    #loss_classifier_act3 = Activation('softmax')(loss3_classifier)
    conv1 = inception_out3
    #inception1 = Conv2D(64,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1) ### 
    #inception1 = MaxPooling2D(pool_size=(1,2))(inception1)
    #inception2 = Conv2D(96,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    #inception2 = Conv2D(64,(1,2),padding="same", activation="relu", W_regularizer=l2(0.0002))(inception2) ### 
    #inception2 = MaxPooling2D(pool_size=(1,2))(inception2)
    inception3 = Conv2D(64,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    inception3 = Conv2D(128,(2,3),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception3) ###
    inception3 = BatchNormalization(axis=-1)(inception3)
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception3)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception3)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception3 = Multiply()([cnn1, cnn2])
    inception3 = MaxPooling2D(pool_size=(1,2))(inception3)
    inception4 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    inception4 = Conv2D(128,(2,4),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception4) ### 
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception4)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception4)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception4 = Multiply()([cnn1, cnn2])    
    inception4 = MaxPooling2D(pool_size=(1,2))(inception4)
    inception5 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    inception5 = Conv2D(128,(2,5),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception5) ### 
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception5)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception5)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception5 = Multiply()([cnn1, cnn2])     
    inception5 = MaxPooling2D(pool_size=(1,2))(inception5)
    inception6 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    inception6 = Conv2D(128,(2,6),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception6) ###
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception6)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception6)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception6 = Multiply()([cnn1, cnn2]) 
    inception6 = MaxPooling2D(pool_size=(1,2))(inception6)
    #inception7 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    #inception7 = Conv2D(64,(1,7),padding="same", activation="relu", W_regularizer=l2(0.0002))(inception7) ### 
    #inception7 = MaxPooling2D(pool_size=(1,2))(inception7)
    #inception8 = Conv2D(16,(1,1), padding="same", activation="relu", W_regularizer=l2(0.0002))(conv1)
    #inception8 = Conv2D(64,(1,8),padding="same", activation="relu", W_regularizer=l2(0.0002))(inception8) ### 
    #inception8 = MaxPooling2D(pool_size=(1,2))(inception8)            
    inception_max    = MaxPooling2D(pool_size=(1,2))(conv1)
    inception_max = Conv2D(128,(1,1),padding="same", activation="linear", W_regularizer=l2(0.0002))(inception_max)
    cnn1       = Lambda(slice1, output_shape=slice1_output_shape)(inception_max)
    cnn2       = Lambda(slice2, output_shape=slice2_output_shape)(inception_max)
    cnn1       = Activation('linear')(cnn1)
    cnn2       = Activation('sigmoid')(cnn2)
    inception_max = Multiply()([cnn1, cnn2])     
    inception_out4 = merge([inception3,inception4,inception5,inception6,inception_max],mode="concat",concat_axis=3)   # N, 240, 32, 256
      ## First link to the output
    #loss_ave_pool3 = AveragePooling2D(pool_size=(5,5),strides=(3,3))(inception_out3)
    #loss3_conv = Conv2D(128,(1,1),padding='same',activation='relu',W_regularizer=l2(0.0002))(loss_ave_pool3)  
    #loss3_flat = Flatten()(loss3_conv)
    #loss3_fc = Dense(1024,activation='relu',W_regularizer=l2(0.0002))(loss3_flat)
    #loss3_drop_fc = Dropout(0.7)(loss3_fc)
    #loss3_classifier = Dense(17,W_regularizer=l2(0.0002))(loss3_drop_fc)
    #loss_classifier_act3 = Activation('softmax')(loss3_classifier)
    '''
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 32, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 16, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 4, 128)
    '''
    a1 = inception_out4
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 240, 1, 256)  
    
    a1 = Reshape((240, 256))(a1) # (N, 240, 256)
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    #model = Model(input=input_logmel, output=out)    
    model = Model(input=input_logmel, output=[loss_classifier_act1,loss_classifier_act2,out])
    model.summary()
    
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    
    # Save model callback
    filepath = os.path.join(args.out_model_dir, "gatedAct_rationBal44_lr0.001_normalization_at_cnnRNN_64newMel_240fr.{epoch:02d}-{val_loss:.4f}.hdf5")
    create_folder(os.path.dirname(filepath))
    save_model = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc', 
                                 verbose=0,
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)  

    # Data generator
    gen = RatioDataGenerator(batch_size=10, type='train')

    # Train
    
    model.fit_generator(generator=gen.generate([tr_x], [tr_y]), 
                        steps_per_epoch=100,    # 100 iters is called an 'epoch'
                        epochs=31,              # Maximum 'epoch' to train
                        verbose=1, 
                        callbacks=[save_model], 
                        validation_data=(te_x, [te_y,te_y,te_y]))
    
    
    #model.fit(x=tr_x,y=tr_y,batch_size=10 ,epochs=2,verbose=1,callbacks=[save_model],validation_data=(te_x, te_y))    

# Run function in mini-batch to save memory. 
def run_func(func, x, batch_size):
    pred_all = []
    batch_num = int(np.ceil(len(x) / float(batch_size)))
    for i1 in xrange(batch_num):
        batch_x = x[batch_size * i1 : batch_size * (i1 + 1)]
        [preds] = func([batch_x, 0.])
        pred_all.append(preds)
    pred_all = np.concatenate(pred_all, axis=0)
    return pred_all

# Recognize and write probabilites. 
def recognize(args, at_bool, sed_bool):
    args.te_hdf5_path = "/data/users/21799506/Data/DCASE2017_Task4/packed_features/logmel/evaluation.h5"
    args.scaler_path="/data/users/21799506/Data/DCASE2017_Task4/scalers/logmel/training.scaler" 
    args.model_dir="/data/users/21799506/Data/DCASE2017_Task4/models/crnn_sed_inception_3to6_gated_v2" 
    args.out_dir="/data/users/21799506/Data/DCASE2017_Task4/preds/crnn_sed_inception_3to6_gated_v2"
    (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    x = te_x
    y = te_y
    na_list = te_na_list
    
    x = do_scale(x, args.scaler_path, verbose=1)
    
    fusion_at_list = []
    fusion_sed_list = []
    for epoch in range(21, 31, 1):    
        t1 = time.time()
        print(epoch)        
        [model_path] = glob.glob(os.path.join(args.model_dir, 
            "*.%02d-0.*.hdf5" % epoch))
        print(model_path)
        print(model_path)
        print(model_path)
        model = load_model(model_path)
        
        # Audio tagging
        if at_bool:
            pred = model.predict(x)
            pred = np.asarray(pred)
            pred = pred[2,:,:]
            print(pred.shape)
            pred = pred.tolist()
            fusion_at_list.append(pred)
        
        # Sound event detection
        if sed_bool:
            in_layer = model.get_layer('in_layer')
            loc_layer = model.get_layer('localization_layer')
            func = K.function([in_layer.input, K.learning_phase()], 
                              [loc_layer.output])
            pred3d = run_func(func, x, batch_size=20)           
            fusion_sed_list.append(pred3d)
        
        print("Prediction time: %s" % (time.time() - t1,))
        
    
    # Write out AT probabilities
    if at_bool:
        fusion_at = np.mean(np.array(fusion_at_list), axis=0)
        print("AT shape: %s" % (fusion_at.shape,))
        io_task4.at_write_prob_mat_to_csv(
            na_list=na_list, 
            prob_mat=fusion_at, 
            out_path=os.path.join(args.out_dir, "at_prob_mat.csv.gz"))
    
    # Write out SED probabilites
    if sed_bool:
        fusion_sed = np.mean(np.array(fusion_sed_list), axis=0)
        #print("SED shape:%s" % (fusion_sed.shape,))
        #name = '/data/users/21799506/Data/DCASE2017_Task4/preds/crnn_sed_train_1/InstanceProb.mat'
        #sio.savemat(name,{'InstanceLabel':fusion_sed})
        
        io_task4.sed_write_prob_mat_list_to_csv(
            na_list=na_list, 
            prob_mat_list=fusion_sed, 
            out_path=os.path.join(args.out_dir, "sed_prob_mat_list.csv.gz"))
        
    print("Prediction finished!")

# Get stats from probabilites. 
def get_stat(args, at_bool, sed_bool):
    lbs = cfg.lbs
    step_time_in_sec = cfg.step_time_in_sec
    max_len = cfg.max_len
    thres_ary = [0.3] * len(lbs)
    args.pred_dir = "/data/users/21799506/Data/DCASE2017_Task4/preds/crnn_sed"
    args.stat_dir = "/data/users/21799506/Data/DCASE2017_Task4/stats/crnn_sed"
    args.submission_dir = "/data/users/21799506/Data/DCASE2017_Task4/submission/crnn_sed"
    # Calculate AT stat
    if at_bool:
        pd_prob_mat_csv_path = os.path.join(args.pred_dir, "at_prob_mat.csv.gz")
        at_stat_path = os.path.join(args.stat_dir, "at_stat.csv")
        at_submission_path = os.path.join(args.submission_dir, "at_submission.csv")
        
        at_evaluator = evaluate.AudioTaggingEvaluate(
            weak_gt_csv="meta_data/groundtruth_weak_label_testing_set.csv", 
            lbs=lbs)
        
        at_stat = at_evaluator.get_stats_from_prob_mat_csv(
                        pd_prob_mat_csv=pd_prob_mat_csv_path, 
                        thres_ary=thres_ary)
                        
        # Write out & print AT stat
        at_evaluator.write_stat_to_csv(stat=at_stat, 
                                       stat_path=at_stat_path)
        at_evaluator.print_stat(stat_path=at_stat_path)
        
        # Write AT to submission format
        io_task4.at_write_prob_mat_csv_to_submission_csv(
            at_prob_mat_path=pd_prob_mat_csv_path, 
            lbs=lbs, 
            thres_ary=at_stat['thres_ary'], 
            out_path=at_submission_path)
               
    # Calculate SED stat
    if sed_bool:
        sed_prob_mat_list_path = os.path.join(args.pred_dir, "sed_prob_mat_list.csv.gz")
        sed_stat_path = os.path.join(args.stat_dir, "sed_stat.csv")
        sed_submission_path = os.path.join(args.submission_dir, "sed_submission.csv")
        
        sed_evaluator = evaluate.SoundEventDetectionEvaluate(
            strong_gt_csv="meta_data/groundtruth_strong_label_testing_set.csv", 
            lbs=lbs, 
            step_sec=step_time_in_sec, 
            max_len=max_len)
                            
        # Write out & print SED stat
        sed_stat = sed_evaluator.get_stats_from_prob_mat_list_csv(
                    pd_prob_mat_list_csv=sed_prob_mat_list_path, 
                    thres_ary=thres_ary)
                    
        # Write SED to submission format
        sed_evaluator.write_stat_to_csv(stat=sed_stat, 
                                        stat_path=sed_stat_path)                     
        sed_evaluator.print_stat(stat_path=sed_stat_path)
        
        # Write SED to submission format
        io_task4.sed_write_prob_mat_list_csv_to_submission_csv(
            sed_prob_mat_list_path=sed_prob_mat_list_path, 
            lbs=lbs, 
            thres_ary=thres_ary, 
            step_sec=step_time_in_sec, 
            out_path=sed_submission_path)
                                                        
    print("Calculating stat finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--tr_hdf5_path', type=str)
    parser_train.add_argument('--te_hdf5_path', type=str)
    parser_train.add_argument('--scaler_path', type=str)
    parser_train.add_argument('--out_model_dir', type=str)
    
    parser_recognize = subparsers.add_parser('recognize')
    parser_recognize.add_argument('--te_hdf5_path', type=str)
    parser_recognize.add_argument('--scaler_path', type=str)
    parser_recognize.add_argument('--model_dir', type=str)
    parser_recognize.add_argument('--out_dir', type=str)
    
    parser_get_stat = subparsers.add_parser('get_stat')
    parser_get_stat.add_argument('--pred_dir', type=str)
    parser_get_stat.add_argument('--stat_dir', type=str)
    parser_get_stat.add_argument('--submission_dir', type=str)
    
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'recognize':
        recognize(args, at_bool=True, sed_bool=True)
    elif args.mode == 'get_stat':
        get_stat(args, at_bool=True, sed_bool=True)
    else:
        raise Exception("Incorrect argument!")
