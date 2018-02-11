import cv2
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model

import sys
import os
import shutil

#Multiple heads and fixed length char sequence

#params
list_a= ['0','1','2','3','4','5','6','7','8','9']
list_b= ['A','B','C']
dx= 20
dy= 50
font_scale= 1
batch_size= 32
epochs= 60
patience = 10
model_name= 'model.h5'
input_dim= (80,120,3)

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
    
    y1= np.zeros((len(list_b)), np.int32)
    y2= np.zeros((len(list_b)), np.int32)
    y3= np.zeros((len(list_a)), np.int32)
    y4= np.zeros((len(list_a)), np.int32)
    y5= np.zeros((len(list_a)), np.int32)
    y6= np.zeros((len(list_b)), np.int32)
    
    y1[list_b.index(letter_str1)]= 1
    y2[list_b.index(letter_str2)]= 1
    y3[list_a.index(number_str1)]= 1
    y4[list_a.index(number_str2)]= 1
    y5[list_a.index(number_str3)]= 1
    y6[list_b.index(letter_str3)]= 1
        
    return img,y1,y2,y3,y4,y5,y6

def batch_generator(batch_size):
    while True:
        image_list = []
        y1_list = []
        y2_list = []
        y3_list = []
        y4_list = []
        y5_list = []
        y6_list = []
        for i in range(batch_size):
            img,y1,y2,y3,y4,y5,y6 = generate_random_plate_number()
            image_list.append(img)
            y1_list.append(y1)
            y2_list.append(y2)
            y3_list.append(y3)
            y4_list.append(y4)
            y5_list.append(y5)
            y6_list.append(y6)
            
        image_arr = np.array(image_list, dtype=np.float32)
        y1_arr = np.array(y1_list, dtype=np.int32)
        y2_arr = np.array(y2_list, dtype=np.int32)
        y3_arr = np.array(y3_list, dtype=np.int32)
        y4_arr = np.array(y4_list, dtype=np.int32)
        y5_arr = np.array(y5_list, dtype=np.int32)
        y6_arr = np.array(y6_list, dtype=np.int32)
                
        yield(image_arr, {'output1': y1_arr, 'output2': y2_arr, 'output3': y3_arr, 'output4': y4_arr, 'output5': y5_arr, 'output6': y6_arr} )
         
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
    
    tail= GlobalAveragePooling2D()(act4)
    
    #add heads
    head1= Dense(len(list_b))(tail)
    softmax1= Activation('softmax', name='output1')(head1)
    head2= Dense(len(list_b))(tail)
    softmax2= Activation('softmax', name='output2')(head2)
    head3= Dense(len(list_a))(tail)
    softmax3= Activation('softmax', name='output3')(head3)
    head4= Dense(len(list_a))(tail)
    softmax4= Activation('softmax', name='output4')(head4)
    head5= Dense(len(list_a))(tail)
    softmax5= Activation('softmax', name='output5')(head5)
    head6= Dense(len(list_b))(tail)
    softmax6= Activation('softmax', name='output6')(head6)
    
    model = Model(input = network_input, output = [softmax1,softmax2,softmax3,softmax4,softmax5,softmax6])
  
    model.compile(optimizer = Adadelta(),
        loss = {'output1': 'categorical_crossentropy', 'output2': 'categorical_crossentropy', 'output3': 'categorical_crossentropy',
        'output4': 'categorical_crossentropy', 'output5': 'categorical_crossentropy', 'output6': 'categorical_crossentropy'})
    
    #print(model.summary()) #
    plot_model(model, to_file='model.png', show_shapes=True) #
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
    
    for i in range(0,len(y_pred)): 
        indx= np.argmax(y_pred[i])
        
        if(y_pred[i].shape[1] == len(list_a)):
            char_str.append(list_a[indx])
        elif(y_pred[i].shape[1] == len(list_b)):
            char_str.append(list_b[indx])
        else:
            print('Output have no mapping!')
            sys.exit()
        
    print('Prediction:', char_str)
    
def create_samples(n_samples):

    if(os.path.exists('generated_samples')):
        shutil.rmtree('generated_samples')
    os.makedirs('generated_samples')
    
    for i in range(0,n_samples):
        img,y1,y2,y3,y4,y5,y6= generate_random_plate_number()
        cv2.imwrite(os.path.join('generated_samples','sample_'+str(i)+'.png'),img)
    
def evaluate_model():
    create_samples(3)
    
    model= get_model()
    model.load_weights(model_name)
    
    for filename in sorted(os.listdir('generated_samples')):
        
        #model= get_model()
        #model.load_weights(model_name)
    
        image_path= os.path.join('generated_samples',filename)
        print('image_path',image_path)
        
        img= cv2.imread(image_path)
        
        y_pred= model.predict(img[None,...])
        
        decode_prediction(y_pred)
    

###########################################################################################
#train()
evaluate_model()
