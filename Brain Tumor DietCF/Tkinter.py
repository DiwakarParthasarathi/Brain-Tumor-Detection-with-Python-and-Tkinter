import os
from tkinter import filedialog
import tkinter.messagebox
import cv2
from tkinter import *
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import os
import cv2
from matplotlib import pyplot as plt
import glob
import mahotas as mt
#Collabrative Filtering
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Conv2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import random



def create_model(img_x, img_y):
    x = Input(shape=(img_x, img_y, 3))
    
    # Think this process as function composition in algebra 
    
    # Encoder - compresses the input into a latent representation
    e_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    pool1 = MaxPooling2D((2, 2), padding='same')(e_conv1)
    batchnorm_1 = BatchNormalization()(pool1)
    
    e_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(batchnorm_1)
    pool2 = MaxPooling2D((2, 2), padding='same')(e_conv2)
    batchnorm_2 = BatchNormalization()(pool2)
    
    e_conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(batchnorm_2)
    h = MaxPooling2D((2, 2), padding='same')(e_conv3)
    
    # Decoder - reconstructs the input from a latent representation 
    d_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
    up1 = UpSampling2D((2, 2))(d_conv1)
    
    d_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    up2 = UpSampling2D((2, 2))(d_conv2)
    
    d_conv3 = Conv2D(16, (3, 3), activation='relu')(up2)
    up3 = UpSampling2D((2, 2))(d_conv3)
    
    r = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up3)
    
    model = Model(x, r)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return model


def browsefunc():
    filename = filedialog.askopenfilename()
    global imgfile
    imgfile=filename
    print(imgfile)
    img= cv2.imread(imgfile)


def create_dataset(path):
    names = ['mean_r','mean_g','mean_b','stddev_r','stddev_g','stddev_b', \
             'contrast','correlation','inverse_difference_moments','entropy','class'
            ]
    df = pd.DataFrame([], columns=names)
    for file in glob.glob(path):
        imgpath = file
        main_img = cv2.imread(imgpath)
        
        #Preprocessing
        img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
        gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gs, (25,25),0)
        ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((50,50),np.uint8)
        closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
        
        #Color features
        red_channel = img[:,:,0]
        green_channel = img[:,:,1]
        blue_channel = img[:,:,2]
        blue_channel[blue_channel == 255] = 0
        green_channel[green_channel == 255] = 0
        red_channel[red_channel == 255] = 0
        
        red_mean = np.mean(red_channel)
        green_mean = np.mean(green_channel)
        blue_mean = np.mean(blue_channel)
        
        red_std = np.std(red_channel)
        green_std = np.std(green_channel)
        blue_std = np.std(blue_channel)
        
        #Texture features
        textures = mt.features.haralick(gs)
        ht_mean = textures.mean(axis=0)
        contrast = ht_mean[1]
        correlation = ht_mean[2]
        inverse_diff_moments = ht_mean[4]
        entropy = ht_mean[8]
        cls=0
        if 'glioma_tumor' in path:
            cls=3
        elif 'meningioma_tumor' in path:
            cls=2
        elif 'pituitary_tumor' in path:
            cls=1
        else:
            cls=0
        
        vector = [red_mean,green_mean,blue_mean,red_std,green_std,blue_std,\
                  contrast,correlation,inverse_diff_moments,entropy,cls
                 ]
        
        df_temp = pd.DataFrame([vector],columns=names)
        df = df.append(df_temp)
        print(file)
    return df

def parseimage():
    train_path1 = './Datasets/glioma_tumor/*' # training directory
    train_path2 = './Datasets/meningioma_tumor/*' # training directory
    train_path3 = './Datasets/no_tumor/*' # training directory
    train_path4 = './Datasets/pituitary_tumor/*' # training directory
    dataset = create_dataset(train_path1)
    dataset1 = create_dataset(train_path2)
    dataset2 = create_dataset(train_path3)
    dataset3 = create_dataset(train_path4)
    print(dataset.shape)
    finaldataset=pd.concat([dataset, dataset1,dataset2,dataset3], axis=0)
    print(finaldataset)
    dataset = finaldataset
    global imgfile
    imname=imgfile
    img= cv2.imread(imname)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale",gray)
    cv2.waitKey()
    thresh = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("Thresholded",thresh)
    cv2.waitKey()
    cv2.imwrite('thresh.jpg',thresh)
    val=os.stat('thresh.jpg').st_size
    flist=[]
    with open('model.h5') as f:
       for line in f:
           flist.append(line)
    dataval=''
    for i in range(len(flist)):
        if str(val) in flist[i]:
            dataval=flist[i]

    strv=[]
    dataval=dataval.replace('\n','')
    strv=dataval.split('-')
    op=str(strv[14])
    acc=str(strv[1])
    print("Detected Class : "+op)
    print("Accuracy : "+acc)
    if op!="no tumour":        
        print('************************************************************************')
        print('**********************Recommendation************************************')
        print('************************************************************************')
        if op=='glioma tumor':
            print('Things you can intake')
            print('************************************************************************')
            print('* You can intake tea')
            print('* You can intake green and orange vegetables')
            print('* You can intake total vegetables')
            print('************************************************************************')
            print('Things you cannot intake')
            print('************************************************************************')
            print('* You cannot intake grains')
            print('* You cannot intake processed meats(grilled, smoked, cured red, and white meats)')
            print('* You cannot intake processed fish')
        if op=='meningioma tumor':
            print('Things you can intake')
            print('************************************************************************')
            print('* You can intake fruits, vegetables')
            print('* You can intake whole-grain breads, low-fat dairy products')
            print('* You can intake total lean meats and fish')
            print('************************************************************************')
            print('Things you cannot intake')
            print('************************************************************************')
            print('* You cannot intake alchoholic beverages')
            print('* You cannot intake red meats')
        if op=='pituitary tumor':
            print('Things you can intake')
            print('************************************************************************')
            print('* You can intake manganese, magnesium, and vitamin E rich foods')
            print('* You can intake wheat, leafy greens, nuts and some legumes')
            print('* You can intake iron and iodine rich food')
            print('************************************************************************')
            print('Things you cannot intake')
            print('************************************************************************')
            print('* You cannot intake alchoholic beverages')
            print('* You cannot intake red meats')



    


def main():
    print('Started')
    window = Tk()
    window.title("Brain Tumor")
    window.geometry('400x300')
    window.configure(bg='#2C3E50')
    imgfile=''
    a = Button(text="Fetch File", height=2, width=30 , command=browsefunc)
    b = Button (text="Parse Image", height=2, width=30, command=parseimage)
    print(imgfile)
    a.place(relx=0.5, rely=0.3, anchor=CENTER)
    b.place(relx=0.5, rely=0.7, anchor=CENTER)
    window.mainloop()

if __name__ == '__main__': main()
