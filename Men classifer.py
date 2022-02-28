#Importing the librabries
import os,cv2
from os import listdir
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from tqdm import tqdm
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

#Importing the data
data_dir = os.path.join('F:/', 'Work', 'Projects', 'Graduation project', 'The classifier', 'Data', 'Men')

categories = ['jackets_top_casual_men_clothes', 'jackets_top_formal_men_clothes',
              'jackets_top_sport_men_clothes','shirts_top_casual_men_clothes',
              'shirts_top_formal_men_clothes', 'shorts_bottom_casual_men_clothes',
              'shorts_bottom_sport_men_clothes', 'suits_sport_men_clothes',
              'sweatshirt_top_casual_men_clothes', 't_shirts_top_casual_men Clothes',
              't_shirts_top_casual_men Clothes', 'trousers_bottom_casual_men clothes',
              'trousers_bottom_formal_men_clothes', 'trousers_bottom_sport_men_clothes']

#Show with the garyscale
for category in categories:
    path = os.path.join(data_dir, category)
    for img in os.listdir(path):
        img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_arr, cmap = 'gray')
        plt.show()
        break
    break

#Image size
img_size = 50
new_arr = cv2.resize(img_arr, (img_size, img_size))
plt.imshow(new_arr, cmap = 'gray')
plt.show()

#Training data
training_data = []
def creat_traingin_data():
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_arr = cv2.resize(img_arr, (img_size, img_size))
                training_data.append([new_arr, class_num])
            except Exception as e:
                pass
creat_traingin_data()

#Suffeling the data
random.shuffle(training_data)

#Test
for sample in training_data[:30]:
    print(sample[1])

#Features and labels
x = []
y = []

for features, labels in training_data:
    x.append(features)
    y.append(labels)

x = np.array(x).reshape(-1, img_size, img_size, 1)
y = np.array(y)

#Noramlize the data
x = x/255.0

#Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

#Building the model
model = Sequential()
model.add(Flatten())
model.add(Dense(200, activation="relu"))
model.add(Dense(64, activation="softmax"))


# Compiling the model
model.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics = ["accuracy"])

# Fitting the model
model.fit(x_train, y_train, epochs = 100, batch_size=65, validation_data=(x_test, y_test))

#Test the model
y_pred = model.predict(x_test)

y_classes = [np.argmax(element) for element in y_pred]


def plot_sample(x, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(x[index])
    plt.xlabel(categories[y[index]])

plot_sample(x_test, y_test,1235)