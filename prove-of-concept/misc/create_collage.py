import numpy as np
import cv2

rows=80
cols= 120
dx= 20
dy= 50
font_scale= 1
list_a= ['0','1','2','3','4','5','6','7','8','9']
list_b= ['A','B','C']
min_number_of_chars= 2
max_number_of_chars= 6
collage_rows= 8
collage_cols= 8

def generate_random_plate_number_fixed_length():
    img= np.zeros((rows,cols,3), np.uint8)
    
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
    
    return img
    
def generate_random_plate_number_variable_length():
    img= np.zeros((rows,cols,3), np.uint8)
    
    n_chars_in_sequence= np.random.randint(min_number_of_chars,max_number_of_chars+1) #[min_number_of_chars,max_number_of_chars]
    
    generated_char_list=[]
    for i in range(0,n_chars_in_sequence):
        if(np.random.uniform() > 0.5):
            rand_indx_a= np.random.randint(len(list_a))
            number_str= list_a[rand_indx_a]
            cv2.putText(img, number_str, (i*dx,dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
            generated_char_list.append(number_str)
        else:
            rand_indx_b= np.random.randint(len(list_b))
            letter_str= list_b[rand_indx_b]
            cv2.putText(img, letter_str, (i*dx,dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=2)
            generated_char_list.append(letter_str)
            
    return img
    
def generate_fixed_length_collage():
    collage= np.zeros((collage_rows*rows,collage_cols*cols,3), np.uint8)
    
    for i in range(0,collage_rows):
        for j in range(0,collage_cols):
            collage[i*rows:(i+1)*rows,j*cols:(j+1)*cols]= generate_random_plate_number_fixed_length()
            cv2.rectangle(collage,(j*cols,i*rows),((j+1)*cols,(i+1)*rows),(0,0,255),3)
            
    cv2.imwrite('fixed_length_samples.png',collage)
    
def generate_variable_length_collage():
    collage= np.zeros((collage_rows*rows,collage_cols*cols,3), np.uint8)
    
    for i in range(0,collage_rows):
        for j in range(0,collage_cols):
            collage[i*rows:(i+1)*rows,j*cols:(j+1)*cols]= generate_random_plate_number_variable_length()
            cv2.rectangle(collage,(j*cols,i*rows),((j+1)*cols,(i+1)*rows),(0,0,255),3)
            
    cv2.imwrite('variable_length_samples.png',collage)

##############################################################################################################################
generate_fixed_length_collage()
generate_variable_length_collage()
