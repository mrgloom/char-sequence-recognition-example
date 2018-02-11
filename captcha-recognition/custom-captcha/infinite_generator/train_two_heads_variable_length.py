import sys
import os
import glob
import shutil
import itertools
import random as rn
import argparse

import cv2
import numpy as np
import pandas as pd

import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from keras import layers
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model

#Two heads and variable length char sequence

list_a = ['0','1','2','3','4','5','6','7','8','9']
list_b = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
list_chars = list_a+list_b
char_dict_length = int(len(list_chars))
min_number_of_chars = 4
max_number_of_chars = 5

input_dim = (80,120,3)
batch_size = 32
n_epochs = 1000*1000
patience = 10
model_name = 'model.h5'

dx = 20
dy = 50
font_scale = 1

#np.random.seed(0)

def generate_sample_v1():
	'''
	Fixed char position, font, size.
	Variable char sequence length.
	'''
	
	img = np.zeros(input_dim, np.uint8)

	# Number of chars in range [min_number_of_chars,max_number_of_chars]
	n_chars_in_sequence = np.random.randint(min_number_of_chars,max_number_of_chars+1)

	generated_char_list = []
	for i in range(0,n_chars_in_sequence):
		if(np.random.uniform() > 0.5):
			rand_indx_a = np.random.randint(len(list_a))
			number_str = list_a[rand_indx_a]
			cv2.putText(img, number_str, (i*dx,dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
			generated_char_list.append(number_str)
		else:
			rand_indx_b = np.random.randint(len(list_b))
			letter_str = list_b[rand_indx_b]
			cv2.putText(img, letter_str, (i*dx,dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
			generated_char_list.append(letter_str)
	
	return img, generated_char_list
	
def generate_sample_v2():
	'''
	Same as v1, but whole string can move in x direction
	Fixed char position, font, size.
	Variable char sequence length.
	'''
	
	img = np.zeros(input_dim, np.uint8)

	# Number of chars in range [min_number_of_chars,max_number_of_chars]
	n_chars_in_sequence = np.random.randint(min_number_of_chars,max_number_of_chars+1)
	
	str_dx = np.random.randint(0,dx//2+1)
	
	generated_char_list = []
	for i in range(0,n_chars_in_sequence):
		if(np.random.uniform() > 0.5):
			rand_indx_a = np.random.randint(len(list_a))
			number_str = list_a[rand_indx_a]
			cv2.putText(img, number_str, (str_dx+i*dx,dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
			generated_char_list.append(number_str)
		else:
			rand_indx_b = np.random.randint(len(list_b))
			letter_str = list_b[rand_indx_b]
			cv2.putText(img, letter_str, (str_dx+i*dx,dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
			generated_char_list.append(letter_str)
	
	return img, generated_char_list

def generate_sample_v3():
	'''
	Same as v2, but whole string can move in x,y direction
	Fixed char position, font, size.
	Variable char sequence length.
	'''
	
	img = np.zeros(input_dim, np.uint8)

	# Number of chars in range [min_number_of_chars,max_number_of_chars]
	n_chars_in_sequence = np.random.randint(min_number_of_chars,max_number_of_chars+1)
	
	str_dx = np.random.randint(0,dx//2+1)
	str_dy = np.random.randint(-10,10+1)
	
	generated_char_list = []
	for i in range(0,n_chars_in_sequence):
		if(np.random.uniform() > 0.5):
			rand_indx_a = np.random.randint(len(list_a))
			number_str = list_a[rand_indx_a]
			cv2.putText(img, number_str, (str_dx+i*dx,str_dy+dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
			generated_char_list.append(number_str)
		else:
			rand_indx_b = np.random.randint(len(list_b))
			letter_str = list_b[rand_indx_b]
			cv2.putText(img, letter_str, (str_dx+i*dx,str_dy+dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
			generated_char_list.append(letter_str)
	
	return img, generated_char_list

def generate_sample_v4():
	'''
	Same as v1, but each char can move x,y
	Fixed char position, font, size.
	Variable char sequence length.
	'''
	
	img = np.zeros(input_dim, np.uint8)

	# Number of chars in range [min_number_of_chars,max_number_of_chars]
	n_chars_in_sequence = np.random.randint(min_number_of_chars,max_number_of_chars+1)
	
	generated_char_list = []
	for i in range(0,n_chars_in_sequence):
		char_dx = np.random.randint(-3,3+1)
		char_dy = np.random.randint(-3,3+1)
		if(np.random.uniform() > 0.5):
			rand_indx_a = np.random.randint(len(list_a))
			number_str = list_a[rand_indx_a]
			cv2.putText(img, number_str, (char_dx+i*dx,char_dy+dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
			generated_char_list.append(number_str)
		else:
			rand_indx_b = np.random.randint(len(list_b))
			letter_str = list_b[rand_indx_b]
			cv2.putText(img, letter_str, (char_dx+i*dx,char_dy+dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
			generated_char_list.append(letter_str)
	
	return img, generated_char_list
	
def generate_sample_v5():
	'''
	Like v3+v4
	Fixed char position, font, size.
	Variable char sequence length.
	'''
	
	img = np.zeros(input_dim, np.uint8)

	# Number of chars in range [min_number_of_chars,max_number_of_chars]
	n_chars_in_sequence = np.random.randint(min_number_of_chars,max_number_of_chars+1)
	
	str_dx = np.random.randint(0,dx//2+1)
	str_dy = np.random.randint(-10,10+1)
	
	generated_char_list = []
	for i in range(0,n_chars_in_sequence):
		char_dx = np.random.randint(-3,3+1)
		char_dy = np.random.randint(-3,3+1)
		if(np.random.uniform() > 0.5):
			rand_indx_a = np.random.randint(len(list_a))
			number_str = list_a[rand_indx_a]
			cv2.putText(img, number_str, (str_dx+char_dx+i*dx,str_dy+char_dy+dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
			generated_char_list.append(number_str)
		else:
			rand_indx_b = np.random.randint(len(list_b))
			letter_str = list_b[rand_indx_b]
			cv2.putText(img, letter_str, (str_dx+char_dx+i*dx,str_dy+char_dy+dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
			generated_char_list.append(letter_str)
	
	return img, generated_char_list

def add_white_noise(img):
	density = rn.uniform(0, 0.1)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if rn.random() < density:
				img[i, j, :] = 255
	return img
	
def add_black_noise(img):
	density = rn.uniform(0, 0.1)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if rn.random() < density:
				img[i, j, :] = 0
	return img
	
def generate_sample_v6():
	'''
	Same as v5, but with noise
	Fixed char position, font, size.
	Variable char sequence length.
	'''
	
	img = np.zeros(input_dim, np.uint8)

	# Number of chars in range [min_number_of_chars,max_number_of_chars]
	n_chars_in_sequence = np.random.randint(min_number_of_chars,max_number_of_chars+1)
	
	str_dx = np.random.randint(0,dx//2+1)
	str_dy = np.random.randint(-10,10+1)
	
	generated_char_list = []
	for i in range(0,n_chars_in_sequence):
		char_dx = np.random.randint(-3,3+1)
		char_dy = np.random.randint(-3,3+1)
		if(np.random.uniform() > 0.5):
			rand_indx_a = np.random.randint(len(list_a))
			number_str = list_a[rand_indx_a]
			cv2.putText(img, number_str, (str_dx+char_dx+i*dx,str_dy+char_dy+dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
			generated_char_list.append(number_str)
		else:
			rand_indx_b = np.random.randint(len(list_b))
			letter_str = list_b[rand_indx_b]
			cv2.putText(img, letter_str, (str_dx+char_dx+i*dx,str_dy+char_dy+dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
			generated_char_list.append(letter_str)
	
	img = add_white_noise(img)
	img = add_black_noise(img)
	
	return img, generated_char_list
	
def get_sample():

	img, generated_char_list = generate_sample_v6()
	
	n_chars_in_sequence = len(generated_char_list)
	
	#Concatenated one-hot encoding of char sequence
	y1 = np.zeros((max_number_of_chars*char_dict_length), np.int32)
	for i in range(0,n_chars_in_sequence):
		y1[i*char_dict_length+list_chars.index(generated_char_list[i])]= 1
	
	#One-hot encoding of number of chars in sequence
	y2 = np.zeros((max_number_of_chars-min_number_of_chars+1), np.int32)
	y2[n_chars_in_sequence-min_number_of_chars]= 1
	
	return img, y1, y2
	
def batch_generator(batch_size):
	while True:
		X = []
		Y1 = []
		Y2 = []
		for _ in range(batch_size):
			img, y1, y2 = get_sample()
			X.append(img)
			Y1.append(y1)
			Y2.append(y2)
			
		yield (np.array(X, dtype=np.float32),
			{'head1': np.array(Y1, dtype=np.int32), 'head2': np.array(Y2, dtype=np.int32)})

def data_generator(n_samples):
	X = []
	Y1 = []
	Y2 = []
	for _ in range(n_samples):
		img, y1, y2 = get_sample()
		X.append(img)
		Y1.append(y1)
		Y2.append(y2)
		
	return (np.array(X, dtype=np.float32),
		{'head1': np.array(Y1, dtype=np.int32), 'head2': np.array(Y2, dtype=np.int32)})
			
def custom_loss_tf(y_true, y_pred): 
	_labels= tf.reshape(y_true[:, :max_number_of_chars * char_dict_length],
		[batch_size, max_number_of_chars, char_dict_length])
	_logits= tf.reshape(y_pred[:, :max_number_of_chars * char_dict_length],
		[batch_size, max_number_of_chars, char_dict_length])

	mask= tf.reduce_sum(_labels, -1) # Sum by last dimension

	#Compute cross entropy loss by last dimension
	loss = tf.nn.softmax_cross_entropy_with_logits(labels= _labels, logits= _logits) 
	loss= loss*mask
	loss = tf.reduce_sum(loss)

	return loss

def cross_entropy_tf(y_true, y_pred):
	loss = tf.nn.softmax_cross_entropy_with_logits(labels= y_true, logits=y_pred)

	return loss

def get_model():
	inputs = Input(shape=input_dim)
	x = BatchNormalization()(inputs)
	
	x = Conv2D(32, (3, 3), padding='same')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(32, (3, 3), padding='same')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(64, (3, 3), padding='same')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = BatchNormalization()(x)
	x =  Activation('relu')(x)

	x = Conv2D(64, (3, 3), padding='same')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x= Flatten()(x)
	tail = Dropout(0.5)(x)
	
	#Add heads
	head1= Dense(max_number_of_chars*char_dict_length, name='head1')(tail)
	head2= Dense(max_number_of_chars-min_number_of_chars+1, name='head2')(tail)

	model = Model(inputs = inputs, outputs = [head1,head2])
	model.compile(optimizer = Adadelta(), loss = {'head1': custom_loss_tf, 'head2': cross_entropy_tf})

	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True)
	sys.exit()

	return model

def train():
	model= get_model()

	callbacks = [
		EarlyStopping(monitor='val_loss', patience=patience, verbose=0)
	]
	
	n_test_samples = 32*4
	X_test,Y_test = data_generator(n_test_samples)
	history = model.fit_generator(
		generator = batch_generator(batch_size),
		steps_per_epoch = 32, # 32*batch_size samples per epoch
		validation_data = (X_test,Y_test),
		epochs = n_epochs,
		shuffle = False,
		callbacks = callbacks)
	
	model.save_weights(model_name)
	pd.DataFrame(history.history).to_csv('train_history.csv', index=False)
	
def decode_prediction(y_pred):
	char_str=[]

	y_pred_head1= y_pred[0]
	y_pred_head2= y_pred[1]

	pred_n_of_chars= np.argmax(y_pred_head2)+min_number_of_chars

	for i in range(0,pred_n_of_chars):
		indx= np.argmax(y_pred_head1[:,i*char_dict_length:(i+1)*char_dict_length])
		char_str.append(list_chars[indx])
		
	return ''.join(char_str)

def evaluate_model():
	
	model = get_model()
	model.load_weights(model_name)
	
	n = 1000
	c = 0
	c_tp = 0
	error_arr = []
	for i in range(0,n):
		img, char_list = generate_sample_v6()
		true_char_str = ''.join(char_list)
		
		y_pred = model.predict(img[np.newaxis,...])
		
		pred_char_str = decode_prediction(y_pred)
		
		if(true_char_str == pred_char_str):
			c_tp += 1
		else:
			error_arr.append([true_char_str,pred_char_str])
		c += 1
	
	print('acc:', 100.0*(c_tp/c), '%', c_tp, '/', c)
	
	'''
	print('-'*60)
	print('analyze errors:')
	for item in error_arr:
		print('true_char_str:', item[0])
		print('pred_char_str:', item[1])
	'''

def create_collage():
	collage_rows = 10
	collage_cols = 10
	rows = 80
	cols = 120
	collage= np.zeros((collage_rows*rows,collage_cols*cols,3), np.uint8)
	
	for i in range(0,collage_rows):
		for j in range(0,collage_cols):
			img,_ = generate_sample_v6()
			collage[i*rows:(i+1)*rows,j*cols:(j+1)*cols]= img
			cv2.rectangle(collage,(j*cols,i*rows),((j+1)*cols,(i+1)*rows),(0,0,255),1)
			
	cv2.imwrite('generated_samples.png',collage)
	
if __name__ == "__main__":
	create_collage()
	
	train()
	evaluate_model()
