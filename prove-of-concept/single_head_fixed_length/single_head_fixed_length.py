import cv2
import numpy as np
import pandas as pd

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.layers.core import Reshape
from keras import layers
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model

import tensorflow as tf

import sys
import os
import shutil

#Single head and fixed length char sequence

#params
list_a= ['0','1','2','3','4','5','6','7','8','9']
list_b= ['A','B','C']
list_chars= list_a+list_b
dx= 20
dy= 50
font_scale= 1
batch_size= 32
epochs= 60
patience = 10
model_name= 'model.h5'
input_dim= (80,120,3)
dict_length= int(len(list_chars))
number_of_chars= 6

def generate_random_plate_number():
    img= np.zeros(input_dim, np.uint8)
     
    rand_indx_b= np.random.randint(len(list_b))
    letter_str1= list_b[rand_indx_b]
    cv2.putText(img, letter_str1, (0,dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
    
    rand_indx_b= np.random.randint(len(list_b))
    letter_str2= list_b[rand_indx_b]
    cv2.putText(img, letter_str2, (dx,dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
    
    rand_indx_a= np.random.randint(len(list_a))
    number_str1= list_a[rand_indx_a]
    cv2.putText(img, number_str1, (2*dx,dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
    
    rand_indx_a= np.random.randint(len(list_a))
    number_str2= list_a[rand_indx_a]
    cv2.putText(img, number_str2, (3*dx,dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
    
    rand_indx_a= np.random.randint(len(list_a))
    number_str3= list_a[rand_indx_a]
    cv2.putText(img, number_str3, (4*dx,dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
    
    rand_indx_b= np.random.randint(len(list_b))
    letter_str3= list_b[rand_indx_b]
    cv2.putText(img, letter_str3, (5*dx,dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
    
    #y is single vector concatenated from one hot encoding representations of each char
    y= np.zeros((number_of_chars*dict_length), np.int32)
    y[list_chars.index(letter_str1)]= 1
    y[dict_length+list_chars.index(letter_str2)]= 1
    y[2*dict_length+list_chars.index(number_str1)]= 1
    y[3*dict_length+list_chars.index(number_str2)]= 1
    y[4*dict_length+list_chars.index(number_str3)]= 1
    y[5*dict_length+list_chars.index(letter_str3)]= 1
        
    return img,y

def batch_generator(batch_size):
    while True:
        image_list = []
        y_list = []
        for i in range(batch_size):
            img,y= generate_random_plate_number()
            image_list.append(img)
            y_list.append(y)
            
        image_arr = np.array(image_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.int32)
        
        yield(image_arr, {'head': y_arr} )

def custom_loss_tf(y_true, y_pred):
    
    #Compute loss by last dimension    
    loss = tf.nn.softmax_cross_entropy_with_logits(labels= tf.reshape(y_true, [batch_size,number_of_chars,dict_length]),
        logits= tf.reshape(y_pred, [batch_size,number_of_chars,dict_length]))
    loss = tf.reduce_sum(loss)
    
    return loss
    
def get_model(): 

    network_input = Input(shape=input_dim)
    
    conv1= Conv2D(32, (3, 3), padding='same')(network_input)
    pool1= MaxPooling2D(pool_size=(2, 2))(conv1)
    act1= Activation('relu')(pool1)
    
    conv2= Conv2D(32, (3, 3), padding='same')(act1)
    pool2= MaxPooling2D(pool_size=(2, 2))(conv2)
    act2= Activation('relu')(pool2)
    
    conv3= Conv2D(64, (3, 3), padding='same')(act2)
    pool3= MaxPooling2D(pool_size=(2, 2))(conv3)
    act3= Activation('relu')(pool3)
    
    conv4= Conv2D(64, (3, 3), padding='same')(act3)
    pool4= MaxPooling2D(pool_size=(2, 2))(conv4)
    act4= Activation('relu')(pool4)
    
    tail= Flatten()(act4)
    
    head= Dense(number_of_chars*dict_length, name='head')(tail)

    model = Model(input = network_input, output = [head])
    model.compile(optimizer = Adadelta(), loss = custom_loss_tf)
    
    #print(model.summary()) #
    #plot_model(model, to_file='model.png', show_shapes=True) #
    #sys.exit() #
    
    return model
    
def train():
    
    model= get_model()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0)
    ]
    
    history = model.fit_generator(
        generator=batch_generator(batch_size),
        nb_epoch=epochs,
        samples_per_epoch=1*batch_size,
        validation_data=batch_generator(batch_size),
        nb_val_samples=1*batch_size,
        verbose=2,
        callbacks=callbacks)

    model.save_weights(model_name)
    pd.DataFrame(history.history).to_csv('train_history.csv', index=False)

def decode_prediction(y_pred):
    char_str=[]
       
    for i in range(0,int(y_pred.shape[0]/dict_length)):
        indx= np.argmax(y_pred[i*dict_length:(i+1)*dict_length])
        char_str.append(list_chars[indx])
        
    print('Prediction:', char_str)
    
def create_samples(n_samples):

    if(os.path.exists('generated_samples')):
        shutil.rmtree('generated_samples')
    os.makedirs('generated_samples')
    
    for i in range(0,n_samples):
        img,y= generate_random_plate_number()
        cv2.imwrite(os.path.join('generated_samples','sample_'+str(i)+'.png'),img)
    
def evaluate_model():
    create_samples(3)

    model= get_model()
    model.load_weights(model_name)
        
    for filename in sorted(os.listdir('generated_samples')):
    
        image_path= os.path.join('generated_samples',filename)
        print('image_path',image_path)
        
        img= cv2.imread(image_path)
        
        y_pred= model.predict(img[None,...])[0]
         
        decode_prediction(y_pred)
    
###########################################################################################
train()
evaluate_model()
