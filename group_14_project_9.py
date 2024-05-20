# %% [markdown]
# # **PACKAGES LOADING :**

# %%
# TENSORFLOW PACKAGE
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.optimizers import Adam

# KERAS PACKAGE
from keras import layers
from keras.layers import Dense,Dropout,Activation,Add,MaxPooling2D,Conv2D,Flatten,BatchNormalization
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras import backend as K

# SKLEARN PACKAGE
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve,cohen_kappa_score

# OTHER PACKAGES
from tqdm import tqdm
from functools import partial
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random
import gc
import json
import math
import scipy
import itertools
import pandas as pd
from PIL import Image
from google.colab.patches import cv2_imshow # to visualize correctly input images
%matplotlib inline

# %% [markdown]
# #**DRIVE SETTINGS**

# %%
from google.colab import drive
drive.mount("/content/drive")
%cd "/content/drive/MyDrive/deep_project"

# %% [markdown]
# # **CONFUSION MATRIX HOME-MADE FUNCTION :**

# %%
def print_confusion_matrix(model, X, y):
    """
    Description:
    ----------
    This function plot the confusion matrix based on the testing data (X_test,Y_test).
    ----------
    Inputs:
    ----------
    model : CNN model used
    X : Target Dataset.
    y : Target Labels.
    Outputs:
    ----------
    conf_matrix:
        Confusion Matrix Plot.
    ----------
    """
    # Model Predictions:
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred,axis=1)
    # Confusion Matrix:
    conf_matrix = confusion_matrix(y, y_pred,normalize="true")
    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(conf_matrix, ax=ax,cmap='Blues', square=True, vmin=0, vmax=1, annot=True, 
                linewidths=.05, fmt=".2f", cbar_kws={"shrink":.8}, 
                xticklabels='auto', yticklabels='auto')
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    return 

# %% [markdown]
# # **KAGGLE DATASET LOADING :**

# %% [markdown]
# **Dataset Loader Function :**

# %%
def Dataset_loader(DIR, RESIZE, sigmaX=10):
    """
    Description:
    ----------
    The function loads the image IMG storaged in the folder DIR and resizes it by RESIZE .

    Inputs:
    ----------
    DIR : Directory Path where the Image is Stored.
    RESIZE : Image Resize Dimension.

    Outputs:
    ----------
    IMG:Loaded Image with the format [RESIZE x RESIZE].
    """
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR,IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":
            img = cv2.imread(PATH)
            img = cv2.resize(img, (RESIZE,RESIZE), 0)           
            IMG.append(np.array(img))
    return IMG

# %% [markdown]
# **Kaggle Dataset Loading :**

# %%
benign_train = np.array(Dataset_loader('/content/drive/MyDrive/deep_project/BreaKHis 400X/train/benign',224))
malign_train = np.array(Dataset_loader('/content/drive/MyDrive/deep_project/BreaKHis 400X/train/malignant',224))
benign_test = np.array(Dataset_loader('/content/drive/MyDrive/deep_project/BreaKHis 400X/test/benign',224))
malign_test = np.array(Dataset_loader('/content/drive/MyDrive/deep_project/BreaKHis 400X/test/malignant',224))

# %% [markdown]
# **Dataset Composition :**

# %%
# TrainingSet 
l_train_b = len(benign_train) # Benign
l_train_m = len(malign_train) # Malign
l_train = l_train_b+l_train_m # Malign + Benign
# TestingSet
l_test_b = len(benign_test) # Benign
l_test_m = len(malign_test) # Malign
l_test = l_test_b+l_test_m # Malign + Benign
# Entire Dataset
l_tot_b = l_test_b + l_train_b # Benign
l_tot_m = l_test_m + l_train_m # Malign
l_tot = l_test+l_train # Malign + Benign

# %% [markdown]
# **Pie Chart Benignant/Malignant Kaggle Dataset :**

# %%
names = [f'Benignant : { l_tot_b}', f'Malingnant : { l_tot_m}']
size = [l_tot_b, l_tot_m]
cmap = plt.get_cmap("tab20c")
colors = cmap(np.arange(3)*4)
# Setting the size of the figure
plt.figure(figsize=(10,10))
# Create a circle at the center of the plot
my_circle = plt.Circle( (0,0), 0.7, color='white')
# Custom wedges
plt.pie(size, labels = names, 
       startangle=90, pctdistance =0.80 ,colors=colors,
       autopct = '%1.1f%%', radius= 1.2, labeldistance=1.05,
       textprops ={ 'fontweight': 'bold','fontsize':14},
       wedgeprops = {'linewidth' : 4, 'edgecolor' : "w" } )
p = plt.gcf()
p.gca().add_artist(my_circle)
# Pie Chart Title
plt.title('Original Dataset Splitting (Benignant vs Malignant)',{'fontweight': 'bold','fontsize':16})
plt.show()

# %% [markdown]
# # **CNN DATASETS GENERATION :**

# %%
# Kaggle Dataset Labels
benign_train_label = np.zeros(len(benign_train)) # 0
malign_train_label = np.ones(len(malign_train)) # 1
benign_test_label = np.zeros(len(benign_test)) # 0
malign_test_label = np.ones(len(malign_test)) # 1
label =  {0:"benignant",1:"malignant"}

# Merge Kaggle Dataset (CNN Datasets)
X_train = np.concatenate((benign_train, malign_train), axis = 0)
X_test = np.concatenate((benign_test, malign_test), axis = 0)

# Merge Kaggle Labels into Categorical (0/1)
y_train = pd.Series(np.concatenate((benign_train_label, malign_train_label), axis = 0),dtype='category')
y_test = pd.Series(np.concatenate((benign_test_label, malign_test_label), axis = 0),dtype='category')

# Shuffle CNN TrainingSet
s1 = np.arange(X_train.shape[0])
np.random.shuffle(s1)
X_train = X_train[s1] # Shuffled TrainingSet
y_train = y_train[s1] # for PIE CHART (0/1)

# Shuffle CNN TestingingSet
s1 = np.arange(X_test.shape[0])
np.random.shuffle(s1)
X_test = X_test[s1] # Shuffled CNN TestingSet
y_test = y_test[s1] # for PIE CHART

# New CNN Training/Validation Set Generation
X_train, X_val, y_train, y_val = train_test_split(
    X_train,y_train,
    test_size = 0.2, # The ValidationSet correspond to the 20% of the CNN TrainingSet
    random_state = 11)

Y_train = to_categorical(y_train, num_classes= 2) # for CNN ([0,1]/[1,0])
Y_val = to_categorical(y_val, num_classes= 2) # for CNN ([0,1]/[1,0])
Y_test = to_categorical(y_test, num_classes= 2) # for CNN ([0,1]/[1,0])

# %% [markdown]
# **VERY IMPORTANT OBSERVATION**: We use y_test/y_val/y_train variables only for graphical purposes and we use Y_test/Y_val/Y_train as elements for the current CNN. We will use this type of nomenclature for the code cells above.

# %% [markdown]
# **CNN Datasets Dimensions:**
# 

# %%
# TrainingSet
l_train_b_new = sum(y_train==0) # Benign
l_train_m_new = sum(y_train==1) # Malign
l_train_new = l_train_b_new + l_train_m_new # Malign + Benign
# ValidationSet
l_val_b_new = sum(y_val==0) # Benign
l_val_m_new = sum(y_val==1) # Malign
l_val_new = l_val_b_new + l_val_m_new # Malign + Benign
# TestingSet
l_test_b_new = sum(y_test==0)  # Benign
l_test_m_new = sum(y_test==1) # Malign
l_test_new = l_test_b_new + l_test_m_new # Malign + Benign
# Entire Dataset
l_tot_b_new = l_test_b_new + l_train_b_new + l_val_b_new # Benign
l_tot_m_new = l_test_m_new + l_train_m_new + l_val_m_new # Malign
l_tot_new = l_test_new + l_train_new + l_val_new# Malign + Benign

# %% [markdown]
# **Nested Pie Chart of CNN Dataset (Without Data Augmentation) :**

# %%
outer_names = [f'Training Set : {l_train_new}', f'Validation Set : {l_val_new}',f'Testing Set : {l_test_new}']
outer_size = [ l_train_new, l_val_new ,l_test_new]
outer_colors = cmap(np.arange(3)*4)
 
inner_names = [f'Benign Train : {l_train_b_new}', f'Malign Train : {l_train_m_new}', 
               f'Benign Val : {l_val_b_new}', f'Malign Val : {l_val_m_new}',
               f'Benign Test : {l_test_b_new}', f'Malign Test : {l_test_m_new}']
inner_size = [l_train_b_new, l_train_m_new, l_val_b_new, l_val_m_new,l_test_b_new,l_test_m_new]
inner_colors = cmap(np.array([1,2,5,6,9,10]))

# Setting the size of the figure
plt.figure(figsize=(10,10))
# Create a circle at the center of the plot
my_circle = plt.Circle( (0,0), 0.25, color='white')

# Plotting the outer pie
plt.pie(outer_size,  labels = outer_names,
       startangle=90, pctdistance =0.80 ,colors=outer_colors,
       autopct = '%1.1f%%', radius= 1.0,labeldistance=1.05,
       textprops ={ 'fontweight': 'bold','fontsize':14},
       wedgeprops = {'linewidth' : 3, 'edgecolor' : "w" } )

# PLotting the inner pie
plt.pie(inner_size, 
        startangle=90, pctdistance =0.55,colors=inner_colors,
        autopct = '%1.1f%%',radius= 0.5,
       textprops ={'fontweight': 'bold' ,'fontsize':12}, 
       wedgeprops = {'linewidth' : 3, 'edgecolor' : "w" } )

# Plotting the pie 
labels = outer_names + inner_names
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Dataset Split With Data Augmentation (Train/Val/Test)',{'fontweight': 'bold','fontsize':16})
plt.legend(labels,loc='lower left', fontsize =14)
plt.show()

# %% [markdown]
# # **CNN STRUCTURE :**

# %%
def build_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    model.add(Conv2D(16, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="same"))
    model.add(Conv2D(32, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="same"))
    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="same"))
    model.add(Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation='softmax'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['accuracy']
    )
    
    return model

K.clear_session()
gc.collect()

resnet = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

model = build_model(resnet ,lr = 1e-4)

# Early stop  to avoid train overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)

# %% [markdown]
# # **1.CONVOLUTIONAL NEURAL NETWORK (NO AUGMENTED DATA) :**

# %% [markdown]
# **Class Balancement :**

# %%
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 classes = np.unique(y_train),
                                                 y = y_train)
class_weights = dict(enumerate(class_weights))
print(class_weights)

# %% [markdown]
# **Model Fit With Class Balancement (Without Aug) :**

# %%
BATCH_SIZE = 32
history = model.fit(
    X_train,Y_train,batch_size = BATCH_SIZE,
    steps_per_epoch=int(X_train.shape[0] / BATCH_SIZE),
    epochs= 9,
    validation_data=(X_val, Y_val),
    callbacks=[early_stop]
)

# %% [markdown]
# **Model Evaluation :**

# %%
test_loss1,test_acc1 =model.evaluate(X_test, Y_test)

# %% [markdown]
# **Model Performance vs Class Unbalancement :**

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

train_acc1 = history.history['accuracy']
val_acc1 = history.history['val_accuracy']
train_loss1 = history.history['loss']
val_loss1 = history.history['val_loss']
epochs = range(1, len(val_loss1) + 1)

# %% [markdown]
# **Accuracy vs Class Balance Table :**

# %%
from prettytable import PrettyTable 
  
# Specify the Column Names while initializing the Table 
myTable = PrettyTable(["Dataset","Best Accuracy"]) 
  
# Add rows 
myTable.add_row(["Training",
                 round(max(train_acc1),3)]) 
myTable.add_row(["Validation", 
                 round(max(val_acc1),3)]) 
myTable.add_row(["Testing",
                 round(test_acc1,3)]) 
  
print(myTable)

# %% [markdown]
# **Accuracy Plot :**

# %%
plt.figure(figsize=(10,5))
plt.plot(epochs, train_acc1, color='b', label='Training Accuracy')
plt.plot(epochs, val_acc1, color='r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# %% [markdown]
# **Loss Plot :**

# %%
plt.figure(figsize=(10,5))
plt.plot(epochs, train_loss1, color='g', label='Training Loss')
plt.plot(epochs, val_loss1, color='y', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# %% [markdown]
# **Confusion Matrix :**

# %%
print_confusion_matrix(model, X_test, y_test)

# %% [markdown]
# # **DATA AUGMENTATION**

# %% [markdown]
# **Dataset Augmentation Transformation :**

# %%
data_gen = ImageDataGenerator(
    zoom_range  = 2, # randomly zoomed images
    rotation_range = 90, # randomly rotate images
    horizontal_flip = True, # randomly flip images
    width_shift_range=.2,
    height_shift_range=.2,
    vertical_flip = True, # randomly flip images
    fill_mode = 'reflect'
    )

# %% [markdown]
# # **CNN STRUCTURE :**

# %%
def build_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    model.add(Conv2D(16, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="same"))
    model.add(Conv2D(32, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="same"))
    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="same"))
    model.add(Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation='softmax'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['accuracy']
    )
    
    return model

K.clear_session()
gc.collect()

resnet = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

model = build_model(resnet ,lr = 1e-4)

# Early stop  to avoid train overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)

# %% [markdown]
# # **2.CONVOLUTIONAL NEURAL NETWORK (WITH DATA AUGMENTATION) :**
# 

# %% [markdown]
# **Class Balancement :**

# %%
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 classes = np.unique(y_train),
                                                 y = y_train)
class_weights = dict(enumerate(class_weights))
print(class_weights)

# %% [markdown]
# **Model Fit With Data Augmentation (Balanced Classes) :**

# %%
BATCH_SIZE = 32

history = model.fit(
    data_gen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=int(X_train.shape[0] / BATCH_SIZE),
    epochs= 9,
    validation_data=(X_val, Y_val),
    callbacks=[early_stop]
)

# %% [markdown]
# 
# **Model Evaluation :**

# %%
test_loss2, test_acc2 = model.evaluate(X_test, Y_test)

# %% [markdown]
# **Model Performance vs Class Unbalancement :**

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

train_acc2 = history.history['accuracy']
val_acc2 = history.history['val_accuracy']
train_loss2 = history.history['loss']
val_loss2 = history.history['val_loss']
epochs = range(1, len(val_loss2) + 1)

# %% [markdown]
# **Accuracy vs Class Balance Table :**

# %%
from prettytable import PrettyTable 
  
# Specify the Column Names while initializing the Table 
myTable = PrettyTable(["Dataset","Best Accuracy"]) 
  
# Add rows 
myTable.add_row(["Training",
                 round(max(train_acc2),3)]) 
myTable.add_row(["Validation",
                 round(max(val_acc2),3)]) 
myTable.add_row(["Testing",
                 round(test_acc2,3)])
  
print(myTable)

# %% [markdown]
# **Accuracy Plot :**

# %%
plt.figure(figsize=(10,5))
plt.plot(epochs, train_acc2, color='b', label='Training Accuracy')
plt.plot(epochs, val_acc2, color='r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# %% [markdown]
# **Loss Plot :**

# %%
plt.figure(figsize=(10,5))
plt.plot(epochs, train_loss2, color='g', label='Training Loss')
plt.plot(epochs, val_loss2, color='y', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# %% [markdown]
# **Confusion Matrix :**

# %%
print_confusion_matrix(model, X_test, y_test)

# %% [markdown]
# # **CNN DATASET STAIN NORMALIZED :**

# %%
#!pip install staintools
#!pip install spams
#import staintools
#import spams

# %% [markdown]
# **Reference Image for Stain Normalization :**

# %%
stain_ref = cv2.imread('/content/drive/MyDrive/deep_project/Stain_ref/stain_1.png')
stain_ref = cv2.resize(stain_ref, (224,224), 0)
cv2_imshow(stain_ref)

# %% [markdown]
# **Stain Normalized Dataset Creation :** 
# A.A. The Code require long running-time to create the stained dataset.

# %%
'''
x_test_stain = []
x_train_stain = []
x_val_stain = []

normalizer = staintools.StainNormalizer(method='vahadane')
normalizer.fit(target)

target = stain_ref
target = staintools.LuminosityStandardizer.standardize(target)
normalizer = staintools.StainNormalizer(method='vahadane')
normalizer.fit(target)

for i in tqdm(range(len(X_train))):
  to_transform = X_train[i]
  to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
  transformed = normalizer.transform(to_transform)
  x_train_stain.append(transformed)
x_train_stain = np.array(x_train_stain)

for i in tqdm(range(len(X_test))):
  to_transform = X_test[i]
  to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
  transformed = normalizer.transform(to_transform)
  x_test_stain.append(transformed)
x_test_stain = np.array(x_test_stain)

for i in tqdm(range(len(X_val))):
  to_transform = X_val[i]
  to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
  transformed = normalizer.transform(to_transform)
  x_val_stain.append(transformed)
x_val_stain = np.array(x_val_stain)'''

# %% [markdown]
# **Stain Normalized Dataset Saving :** In order to reduce time for the staining processing step, we've saved the stained dataset in the current folders:
# *   Stain Normalized TrainigSet Path : 
#   > X_train_stain : "/content/drive/MyDrive/deep_project/BreaKHis 400X/train/x_train_norm.pickle"
# 
#   > Y_train_stain : "/content/drive/MyDrive/deep_project/BreaKHis 400X/train/y_train_norm.pickle"
# 
# *   Stain Normalized TestingSet Path :  
#   > X_test_stain : "/content/drive/MyDrive/deep_project/BreaKHis 400X/test/x_test_norm.pickle"
# 
#   > Y_test_stain : "/content/drive/MyDrive/deep_project/BreaKHis 400X/test/y_test_norm.pickle"
# *   Stain Normalized ValidationSet Path : 
#   > X_val_stain : "/content/drive/MyDrive/deep_project/BreaKHis 400X/train/x_val_norm.pickle"
#   
#   > Y_val_stain : "/content/drive/MyDrive/deep_project/BreaKHis 400X/train/y_val_norm.pickle"
# 
# 

# %%
'''
pickle_out = open("/content/drive/MyDrive/deep_project/BreaKHis 400X/train/x_train_norm.pickle", "wb")
pickle.dump(x_train_stain, pickle_out)
pickle_out.close()
pickle_out = open("/content/drive/MyDrive/deep_project/BreaKHis 400X/train/y_train_norm.pickle", "wb")
pickle.dump(Y_train, pickle_out)
pickle_out.close()

pickle_out = open("/content/drive/MyDrive/deep_project/BreaKHis 400X/test/x_test_norm.pickle", "wb")
pickle.dump(x_test_stain, pickle_out)
pickle_out.close()
pickle_out = open("/content/drive/MyDrive/deep_project/BreaKHis 400X/test/y_test_norm.pickle", "wb")
pickle.dump(Y_test, pickle_out)
pickle_out.close()

pickle_out = open("/content/drive/MyDrive/deep_project/BreaKHis 400X/train/x_val_norm.pickle", "wb")
pickle.dump(x_val_stain, pickle_out)
pickle_out.close()
pickle_out = open("/content/drive/MyDrive/deep_project/BreaKHis 400X/train/y_val_norm.pickle", "wb")
pickle.dump(Y_val, pickle_out)
pickle_out.close()
'''

# %% [markdown]
# **Stain Normalized Dataset Loading :**

# %%
with open("/content/drive/MyDrive/deep_project/BreaKHis 400X/test/x_test_norm.pickle", "rb") as input_file:
  X_test_stain = pickle.load(input_file)
with open("/content/drive/MyDrive/deep_project/BreaKHis 400X/test/y_test_norm.pickle", "rb") as input_file:
  Y_test_stain = pickle.load(input_file) # For the CNN ([0,1]/[1,0])
y_test_stain = np.argmax(Y_test_stain,axis=1) # For the Confusion Matrix (1/0)

with open("/content/drive/MyDrive/deep_project/BreaKHis 400X/train/x_train_norm.pickle", "rb") as input_file:
  X_train_stain = pickle.load(input_file)
with open("/content/drive/MyDrive/deep_project/BreaKHis 400X/train/y_train_norm.pickle", "rb") as input_file:
  Y_train_stain = pickle.load(input_file) # For the CNN ([0,1]/[1,0])
y_train_stain = np.argmax(Y_train_stain,axis=1) # For Classes Weights (1/0)

with open("/content/drive/MyDrive/deep_project/BreaKHis 400X/train/x_val_norm.pickle", "rb") as input_file:
  X_val_stain = pickle.load(input_file)
with open("/content/drive/MyDrive/deep_project/BreaKHis 400X/train/y_val_norm.pickle", "rb") as input_file:
  Y_val_stain = pickle.load(input_file) # For the CNN ([0,1]/[1,0])
y_val_stain = np.argmax(Y_val_stain,axis=1) # For Classes Weights (1/0)

# %% [markdown]
# **Stain Normalized Dataset Dims :**

# %%
# TrainingSet
l_train_b_stain = sum(y_train_stain==0) # Benign
l_train_m_stain = sum(y_train_stain==1) # Malign
l_train_stain = l_train_b_stain + l_train_m_stain # Malign + Benign
# ValidationSet
l_val_b_stain = sum(y_val_stain==0) # Benign
l_val_m_stain = sum(y_val_stain==1) # Malign
l_val_stain = l_val_b_stain + l_val_m_stain # Malign + Benign
# TestingSet
l_test_b_stain = sum(y_test_stain==0) # Benign
l_test_m_stain = sum(y_test_stain==1) # Malign
l_test_stain = l_test_b_stain + l_test_m_stain # Malign + Benign
# Entire Dataset
l_tot_b_stain = l_test_b_stain + l_train_b_stain + l_val_b_stain # Benign
l_tot_m_stain= l_test_m_stain + l_train_m_stain + l_val_m_stain # Malign
l_tot_stain = l_test_stain + l_train_stain + l_val_stain # Malign + Benign

# %% [markdown]
# # **CNN STRUCTURE :**

# %%
def build_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    model.add(Conv2D(16, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="same"))
    model.add(Conv2D(32, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="same"))
    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="same"))
    model.add(Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation='softmax'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['accuracy']
    )
    
    return model

K.clear_session()
gc.collect()

resnet = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

model = build_model(resnet ,lr = 1e-4)

# Early stop  to avoid train overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)

# %% [markdown]
# # **3.CONVOLUTIONAL NEURAL NETWORK (WITH STAIN NORMALIZATION) :**
# 

# %% [markdown]
# **Classes Weights :**

# %%
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 classes = np.unique(y_train_stain),
                                                 y = y_train_stain)
class_weights = dict(enumerate(class_weights))
print(class_weights)

# %% [markdown]
# **Model Fit With Class Balancement :**

# %%
BATCH_SIZE = 32
history = model.fit(
    X_train_stain,Y_train_stain,batch_size = BATCH_SIZE,
    steps_per_epoch=int(X_train_stain.shape[0] / BATCH_SIZE),
    epochs= 9,
    validation_data=(X_val_stain, Y_val_stain),
    class_weight = class_weights,
    callbacks=[early_stop]
)

# %% [markdown]
# **Model Evaluation :**

# %%
test_loss3,test_acc3 = model.evaluate(X_test_stain, Y_test_stain)

# %% [markdown]
# **Model Performance vs Class Unbalancement :**

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

train_acc3 = history.history['accuracy']
val_acc3 = history.history['val_accuracy']
train_loss3 = history.history['loss']
val_loss3 = history.history['val_loss']
epochs = range(1, len(val_loss3) + 1)

# %% [markdown]
# **Accuracy vs Class Balance Table :**

# %%
from prettytable import PrettyTable 
  
# Specify the Column Names while initializing the Table 
myTable = PrettyTable(["Dataset","Best Accuracy"]) 
  
# Add rows 
myTable.add_row(["Training",
                 round(max(train_acc3),3)]) 
myTable.add_row(["Validation",
                 round(max(val_acc3),3)]) 
myTable.add_row(["Testing",
                 round(test_acc3,3)])
  
print(myTable)

# %% [markdown]
# **Accuracy Plot :**

# %%
plt.figure(figsize=(10,5))
plt.plot(epochs, train_acc3, color='b', label='Training Accuracy')
plt.plot(epochs, val_acc3, color='r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# %% [markdown]
# **Loss Plot :**

# %%
plt.figure(figsize=(10,5))
plt.plot(epochs, train_loss3, color='g', label='Training Loss')
plt.plot(epochs, val_loss3, color='y', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# %% [markdown]
# **Confusion Matrix :**

# %%
print_confusion_matrix(model, X_test_stain, y_test_stain)

# %% [markdown]
# # **CNN STRUCTURE :**

# %%
def build_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    model.add(Conv2D(16, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="same"))
    model.add(Conv2D(32, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="same"))
    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="same"))
    model.add(Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation='softmax'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['accuracy']
    )
    
    return model

K.clear_session()
gc.collect()

resnet = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

model = build_model(resnet ,lr = 1e-4)

# Early stop  to avoid train overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)

# %% [markdown]
# # **4.CONVOLUTIONAL NEURAL NETWORK (DATA AUGMENTATION APPLIED TO NORMALIZED DATASET) :**
# 

# %% [markdown]
# **Model Fit With Class Balancement :**

# %%
BATCH_SIZE = 32
history = model.fit(
    data_gen.flow(X_train_stain,Y_train_stain,batch_size = BATCH_SIZE),
    steps_per_epoch=int(X_train_stain.shape[0] / BATCH_SIZE),
    epochs= 9,
    validation_data=(X_val_stain, Y_val_stain),
    callbacks=[early_stop]
)

# %% [markdown]
# **Model Evaluation :**

# %%
test_loss4,test_acc4 = model.evaluate(X_test_stain,Y_test_stain)

# %% [markdown]
# **Model Performance vs Class Unbalancement :**

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

train_acc4 = history.history['accuracy']
val_acc4 = history.history['val_accuracy']
train_loss4 = history.history['loss']
val_loss4 = history.history['val_loss']
epochs = range(1, len(val_loss4) + 1)

# %% [markdown]
# **Accuracy vs Classes Unbalancement:**

# %%
from prettytable import PrettyTable 
  
# Specify the Column Names while initializing the Table 
myTable = PrettyTable(["Dataset", "BestAccuracy"]) 
  
# Add rows 

myTable.add_row(["Training", 
                 round(max(train_acc4),3)]) 
myTable.add_row(["Validation", 
                 round(max(val_acc4),3)]) 
myTable.add_row(["Testing",
                 round(test_acc4,3)])
  
print(myTable)

# %% [markdown]
# **Accuracy Plot :**

# %%
plt.figure(figsize=(10,5))
plt.plot(epochs, train_acc4, color='b', label='Training Accuracy')
plt.plot(epochs, val_acc4, color='r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# %% [markdown]
# **Loss Plot :**

# %%
plt.figure(figsize=(10,5))
plt.plot(epochs, train_loss4, color='g', label='Training Loss')
plt.plot(epochs, val_loss4, color='y', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# %% [markdown]
# **Confusion Matrix :**

# %%
print_confusion_matrix(model, X_test_stain, y_test_stain)

# %% [markdown]
# # **COLOR AUGMENTATION :** 
# From each augmented image we've performed a stain normalization using several type of reference images. With this procedure we should increment the color data variability by using different stains for each image. We apply color data augmentation to the original test data.

# %% [markdown]
# **Stain Reference Images Loading :** 

# %%
!pip install staintools
!pip install spams
import staintools
import spams

# %%
stain_ref = np.array(Dataset_loader('/content/drive/MyDrive/deep_project/Stain_ref',224))

# %% [markdown]
# **Reference Image,Moving Image,Transformed Image Plot:** 

# %%
import staintools
import spams

idx_stain_ref_sel = random.randrange(len(stain_ref))
target = stain_ref[idx_stain_ref_sel] # Random Selection of the Reference Image
target = staintools.LuminosityStandardizer.standardize(target)
normalizer = staintools.StainNormalizer(method='vahadane')
normalizer.fit(target)
to_transform = X_train[random.randrange(len(X_train))] # Random Selection of the Image to Transform 
to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
transformed = normalizer.transform(to_transform) # Transformed Image 
img = cv2.hconcat([target,to_transform,transformed])
print("=================")
print(f"Stain Reference # {idx_stain_ref_sel}")
print(["Reference Image","Moving Image","Normalized Image"])
print("=================")
cv2_imshow(img)

# %% [markdown]
# **Best Stain Reference Images Selection :**
# By manual visual inspection we select the "best" reference images to use for color augmentation.

# %%
good_ref_idx = [0,1,2,4,5,7,8,10,11]

# %% [markdown]
# **Color Augmented Dataset Generation :**

# %%
'''
x_train_col = []
x_val_col = []
y_train_col = []
y_val_col = []

for i in tqdm(range(len(stain_ref))):
  target = stain_ref[i]
  target = staintools.LuminosityStandardizer.standardize(target)
  normalizer = staintools.StainNormalizer(method='vahadane')
  normalizer.fit(target)
  for j in range(160):
    ind = random.randrange(len(X_train_aug))
    to_transform = X_train_aug[ind]
    to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
    transformed = normalizer.transform(to_transform)
    x_train_col.append(transformed)
    y_train_col.append(Y_train_aug[ind])
x_train_col = np.array(x_train_col)
y_train_col = np.array(y_train_col)

for i in tqdm(range(len(stain_ref))):
  target = stain_ref[i]
  target = staintools.LuminosityStandardizer.standardize(target)
  normalizer = staintools.StainNormalizer(method='vahadane')
  normalizer.fit(target)
  for j in range(40):
    ind = random.randrange(len(X_val_aug))
    to_transform = X_val_aug[ind]
    to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
    transformed = normalizer.transform(to_transform)
    x_val_col.append(transformed)
    y_val_col.append(Y_val_aug[ind])
x_val_col = np.array(x_val_col)
y_val_col = np.array(y_val_col)
'''

# %% [markdown]
# **Color Augmented Dataset Saving/Loading Path :**

# %%
# Trainig Set
x_train_col_path = "/content/drive/MyDrive/deep_project/BreaKHis 400X/train/x_train_col.pickle"
y_train_col_path = "/content/drive/MyDrive/deep_project/BreaKHis 400X/train/y_train_col.pickle"
# Validation Set
x_val_col_path = "/content/drive/MyDrive/deep_project/BreaKHis 400X/train/x_val_col.pickle"
y_val_col_path = "/content/drive/MyDrive/deep_project/BreaKHis 400X/train/y_val_col.pickle"
# Validation Set
x_test_col_path = "/content/drive/MyDrive/deep_project/BreaKHis 400X/test/x_test_col.pickle"
y_test_col_path = "/content/drive/MyDrive/deep_project/BreaKHis 400X/test/y_test_col.pickle"

# %% [markdown]
# **Color Augmented Dataset Saving :**

# %%
'''
pickle_out = open("x_train_col_path", "wb")
pickle.dump(x_train_col, pickle_out)
pickle_out.close()
pickle_out = open("y_train_col_path", "wb")
pickle.dump(y_train_col, pickle_out)
pickle_out.close()

pickle_out = open("x_val_col_path", "wb")
pickle.dump(x_val_col, pickle_out)
pickle_out.close()
pickle_out = open("y_val_col_path", "wb")
pickle.dump(y_val_col, pickle_out)
pickle_out.close()

pickle_out = open("x_test_col_path", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()
pickle_out = open("y_test_col_path, "wb")
pickle.dump(Y_test, pickle_out)
pickle_out.close()
'''

# %% [markdown]
# **Color Augmented Dataset Loading :**

# %%
# Trainig Set
with open(x_train_col_path, "rb") as input_file:
  X_train_col = pickle.load(input_file)
with open(y_train_col_path, "rb") as input_file:
  Y_train_col = pickle.load(input_file) # for CNN ([0,1]/[1,0])
y_train_col = np.argmax(Y_train_col,axis=1)
# Validation Set
with open(x_val_col_path, "rb") as input_file:
  X_val_col = pickle.load(input_file)
with open(y_val_col_path, "rb") as input_file:
  Y_val_col = pickle.load(input_file) # for CNN ([0,1]/[1,0])
y_val_col = np.argmax(Y_val_col,axis=1)
# Testing Set
with open(x_test_col_path, "rb") as input_file:
  X_test_col = pickle.load(input_file)
with open(y_test_col_path, "rb") as input_file:
  Y_test_col = pickle.load(input_file) # for CNN ([0,1]/[1,0])
y_test_col = np.argmax(Y_test_col,axis=1)

# %% [markdown]
# **Observation :** The testing set X_test_col/Y_test_col is the same of X_test_new/Y_test_new used for the first CNN. We've name X_test_new/Y_test_new to X_test_col/Y_test_col just for simmetry.

# %% [markdown]
# # **CNN STRUCTURE :**

# %%
def build_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    model.add(Conv2D(16, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="same"))
    model.add(Conv2D(32, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="same"))
    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="same"))
    model.add(Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation='softmax'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['accuracy']
    )
    
    return model

K.clear_session()
gc.collect()

resnet = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

model = build_model(resnet ,lr = 1e-4)

# Early stop  to avoid train overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)

# %% [markdown]
# # **5.CONVOLUTIONAL NEURAL NETWORK (WITH COLOR AUGMENTATION) :**
# 

# %% [markdown]
# **Model Fit with Class Balancement :**

# %%
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 classes = np.unique(y_train_col),
                                                 y = y_train_col)
class_weights = dict(enumerate(class_weights))
print(class_weights)

# %%
BATCH_SIZE = 32
history = model.fit(
    data_gen.flow(X_train_col,Y_train_col,batch_size = BATCH_SIZE),
    steps_per_epoch=int(X_train_col.shape[0] / BATCH_SIZE),
    epochs= 9,
    validation_data=(X_val_col, Y_val_col),
    callbacks=[early_stop],
    class_weight = class_weights
)

# %% [markdown]
# **Model Evaluation :**

# %%
test_loss5,test_acc5 = model.evaluate(X_test_col, Y_test_col)

# %% [markdown]
# **MODEL PERFORMANCE VS CLASS UNBALACEMENT :**

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

train_acc5 = history.history['accuracy']
val_acc5 = history.history['val_accuracy']
train_loss5 = history.history['loss']
val_loss5 = history.history['val_loss']
epochs = range(1, len(val_loss5) + 1)

# %%
from prettytable import PrettyTable 
  
# Specify the Column Names while initializing the Table 
myTable = PrettyTable(["Dataset","Best Accuracy"]) 
  
# Add rows 
myTable.add_row(["Training",
                 round(max(train_acc5),3)]) 
myTable.add_row(["Validation", 
                 round(max(val_acc5),3)]) 
myTable.add_row(["Testing",
                 round(test_acc5,3)])
  
print(myTable)

# %% [markdown]
# **Accuracy Plot :**

# %%
plt.figure(figsize=(10,5))
plt.plot(epochs, train_acc5, color='b', label='Training Accuracy')
plt.plot(epochs, val_acc5, color='r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# %% [markdown]
# **Loss Plot :**

# %%
plt.figure(figsize=(10,5))
plt.plot(epochs, train_loss5, color='g', label='Training Loss')
plt.plot(epochs, val_loss5, color='y', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# **Confusion Matrix:**

# %%
print_confusion_matrix(model, X_test, y_test)

# %% [markdown]
# # **FINAL OBSERVATIONS :**

# %%
from prettytable import PrettyTable  
# Specify the Column Names while initializing the Table 
myTable = PrettyTable(["ID CNN","Data Augmentation",
                       "Stain Normalization","Color Augmentation","Test Loss","Test Accuracy"]) 
# Add rows 
myTable.add_row(["001","No","No","No",round(test_loss1,2),round(test_acc1,2)])
myTable.add_row(["002","Yes","No","No",round(test_loss2,2),round(test_acc2,2)])
myTable.add_row(["003","No","Yes","No",round(test_loss3,2),round(test_acc3,2)])
myTable.add_row(["004","Yes","Yes","No",round(test_loss4,2),round(test_acc4,2)])
myTable.add_row(["004","Yes","No","Yes",round(test_loss5,2),round(test_acc5,2)])
print(myTable)

