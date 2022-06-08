# -*- coding: utf-8 -*-
"""
@author: Adrian Ramos
"""
# conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import os
import numpy as np
import pandas as pd

# Graphical APIs
import SimpleITK
import cv2
import PIL.Image as Image
from skimage.io import (imread,imshow)
from skimage.transform import resize
import matplotlib.pyplot as plt

# Deep Learning imports: 
from unet import Unet
import torch
from torch import (nn, optim)

import tensorflow as tf
from tensorflow import keras as tf_ks
import tensorflow.keras.models
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers 

img_wdt, img_hgt, img_chnls = 128, 128, 3
path = "../Data/Liver_Segmentation-3D-ircadb-01-IRCAD_files/"

# data = pd.read_excel('../Data/dataset_mortalidad_por_enfermedades_no_transmisibles_en_jalisco_durante_el_ano_2015.xlsx')
#%% Dataset extraction
a=cv2.imread("../Data/Liver_Segmentation-3D-ircadb-01-IRCAD_files/liver_13.jpg")
print(a.shape)
print(a)
img_x = Image.open("../Data/Liver_Segmentation-3D-ircadb-01-IRCAD_files/liver_13.jpg")
print(img_x.mode)
print(img_x)
img_x = img_x.convert('RGB')
b = np.array(img_x)
print(b.shape)

img = imread(path +'liver_13'+'.jpg')[:,:,:img_chnls]
imshow(img)
img_resized = resize(img, (img_hgt, img_wdt), mode='constant', preserve_range=False)
fig = imshow(img_resized)

X_train = img_resized

mask = np.zeros((img_hgt, img_wdt, 1), dtype=bool)
Y_train = mask
#%% Model definition

# Hyperparameters:
epochs = 20
learning_rate = 0.001
decay_rate = learning_rate/epochs
omentum = 0.8
batch_size = 20

# # 是否使用cuda
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     device = CUDA_VISIBLE_DEVICES = 3
    
# model = Unet(3,1).to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Contraction Path:

inputs = tf_ks.Input((img_wdt, img_hgt, img_chnls))
s = kl.Lambda(lambda x: x / 255)(inputs)

c1 = kl.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal',padding='same')(s)
c1 = kl.Dropout(0.1)(c1)
c1 = kl.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c1)
mp1 = kl.MaxPooling2D((2,2))(c1)

c2 = kl.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal',padding='same')(mp1)
c2 = kl.Dropout(0.1)(c2)
c2 = kl.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c2)
mp2 = kl.MaxPooling2D((2,2))(c2)

c3 = kl.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal',padding='same')(mp2)
c3 = kl.Dropout(0.2)(c3)
c3 = kl.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c3)
mp3 = kl.MaxPooling2D((2,2))(c3)

c4 = kl.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal',padding='same')(mp3)
c4 = kl.Dropout(0.2)(c4)
c4 = kl.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c4)
mp4 = kl.MaxPooling2D((2,2))(c4)

c5 = kl.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal',padding='same')(mp4)
c5 = kl.Dropout(0.3)(c5)
c5 = kl.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c5)

# Expansive Path
u6 = kl.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
u6 = kl.concatenate([u6,c4])
c6 = kl.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = kl.Dropout(0.2)(c6)
c6 = kl.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = kl.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
u7 = kl.concatenate([u7,c3])
c7 = kl.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = kl.Dropout(0.2)(c7)
c7 = kl.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = kl.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
u8 = kl.concatenate([u8,c2])
c8 = kl.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = kl.Dropout(0.1)(c8)
c8 = kl.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = kl.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
u9 = kl.concatenate([u9,c1], axis=3)
c9 = kl.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = kl.Dropout(0.1)(c8)
c9 = kl.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = kl.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf_ks.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

check_ptr = tf_ks.callbacks.ModelCheckpoint('model_for_liver.h5', verbose=1, save_best_only=True)
callbacks = [tf_ks.callbacks.EarlyStopping(patience=2,monitor='val_loss'),
             tf_ks.callbacks.TensorBoard(log_dir='logs'),
             check_ptr]


# Training
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=batch_size, epochs=epochs, callbacks=callbacks)












