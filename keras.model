import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
from scipy import misc
import cv2
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Merge
from keras.layers import *
from keras.models import Model
import csv
import pickle
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split

# TODO: Fill this in based on where you saved the training and testing data

training_file = "traffic-signs-data/train.p"
testing_file = "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))



print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print(y_train.shape)
img_size = X_train[0].shape[0]
num_channels = X_train[0].shape[2]
num_classes = n_classes

height = X_train[0].shape[0]
width = X_train[0].shape[1]
#read parse csv

def read_csv_and_parse():
    traffic_labels_dict ={}
    with open('signnames.csv') as f:
        reader = csv.reader(f)
        count = -1;
        for row in reader:
            count = count + 1
            if(count == 0):
                continue
            label_index = int(row[0])
            traffic_labels_dict[label_index] = row[1]
    return traffic_labels_dict
traffic_labels_dict = read_csv_and_parse()



# Visualizations will be shown in the notebook.


def get_images_to_plot(images, labels):
    selected_image = []
    idx = []
    for i in range(n_classes):
        selected = np.where(labels == i)[0][0]
        selected_image.append(images[selected])
        idx.append(selected)
    return selected_image, idx



def plot_images(selected_image, row=5, col=10, idx=None):
    count = 0;
    f, axarr = plt.subplots(row, col, figsize=(50, 50))

    for i in range(row):
        for j in range(col):
            if (count < len(selected_image)):
                axarr[i, j].imshow(selected_image[count])
                if (idx != None):
                    axarr[i, j].set_title(traffic_labels_dict[y_train[idx[count]]], fontsize=20)
            axarr[i, j].axis('off')
            count = count + 1
    plt.show()


def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(cls)-1.
    :return:
        2-dim array of shape: [len(cls), num_classes]
    """

    # Find the number of classes if None is provided.
    if num_classes is None:
        num_classes = np.max(class_numbers) - 1

    return np.eye(num_classes, dtype=float)[class_numbers]

def get_additional(count, label, X_train, y_train):
    selected = np.where(y_train == label)[0]
    counter = 0;
    m = 0;
    # just select the first element in selected labels
    X_mqp = X_train[selected[0]]
    X_mqp = X_mqp[np.newaxis, ...]
    while m < (len(selected)):
        # ignore the first element, since it already selected
        aa = X_train[selected[m]]
        X_mqp = np.vstack([X_mqp, aa[np.newaxis, ...]])
        if (counter >= count):
            break
        if (m == (len(selected) - 1)):
            m = 0
        counter = counter + 1
        m = m + 1
    Y_mqp = np.full((len(X_mqp)), label, dtype='uint8')

    return X_mqp, Y_mqp

def balance_dataset(X_train_extra, Y_train_extra):
    hist = np.bincount(y_train)
    max_count = np.max(hist)
    for i in range(len(hist)):
        X_mqp, Y_mqp = get_additional(max_count - hist[i], i, X_train, y_train)
        X_train_extra = np.vstack([X_train_extra, X_mqp])
        Y_train_extra = np.append(Y_train_extra, Y_mqp)
    return X_train_extra,Y_train_extra

X_train_extra,Y_train_extra = X_train,y_train;
print("length",len(y_train))
X_train_extra,Y_train_extra = balance_dataset(X_train_extra,Y_train_extra)

print("length",len(X_train_extra))

Y_train_hot = one_hot_encoded(Y_train_extra,n_classes)
Y_test_hot = one_hot_encoded(y_test,n_classes)
print("Y_train shape",Y_train_hot.shape)

X_train_set,X_validation,Y_train_set,Y_validation = train_test_split( X_train_extra, Y_train_hot, test_size=0.2, random_state=42)
print(X_validation.shape)



inputs = Input(shape=(height,width,num_channels))
lam_layer  = Lambda(lambda x:  x/127.5 - 1.0, name="noramlise")(inputs)
conv_1 =  Convolution2D(24, 3, 3, border_mode='same', activation='elu')(lam_layer)
conv_2 =  Convolution2D(36, 3, 3, border_mode='same', activation='elu')(conv_1)
conv_3 =  Convolution2D(24, 3, 3, border_mode='same', activation='elu')(conv_2)
conv_4 =  Convolution2D(48, 3, 3, border_mode='same', activation='elu')(conv_3)
conv_5 =  Convolution2D(64, 3, 3, border_mode='same', activation='elu')(conv_4)
drop = Dropout(0.5)(conv_5)
flat_1 = Flatten()(drop)
dense = Dense(43)(flat_1)
final = Activation('softmax')(dense)

model = Model(input=inputs, output=final)


batch_size = 64

# I trained for 10 Epoch. I noticed that after 10 epoch the training accuray doesnt decrease further
nb_epoch = 10;

# I used adam optimiser . Mse loss function is used
model.compile('adam', 'categorical_crossentropy',  metrics=['accuracy'])

# a python generator that select images at random with some random augmentation
def generate_train_batch(X_train, y_train, batch_size=32):
    batch_images = np.zeros((batch_size, height, width, 3))
    batch_y_train = np.zeros((batch_size,43))
    while 1:
        bias = 0
        for i_batch in range(batch_size):
            select = np.random.randint(0,len(X_train))
            batch_images[i_batch] = X_train[select]
            batch_y_train[i_batch] = y_train[select]
        yield batch_images, batch_y_train

#  a function to generate random training data of given batch size. This is passed to model.fit_generator function.
train_generator = generate_train_batch(X_train_set, Y_train_set, batch_size)
#the actual training
model.fit_generator(train_generator,
                    samples_per_epoch=40000,
                    nb_epoch=nb_epoch,
                    validation_data=(X_validation, Y_validation)
                    )

#evaluation on test data. Not a useful measure.
result = model.evaluate(X_validation, Y_validation, batch_size=batch_size, verbose=1, sample_weight=None)
print(result)

# save the model
model_filename = "small_model.json";
model_weights = "small_model.h5"
model_json = model.to_json()

# remove old models
try:
    os.remove(model_json)
    os.remove(model_weights)
except OSError:
    pass

#write new model
with open(model_filename, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_weights)
print("Saved model to disk")
