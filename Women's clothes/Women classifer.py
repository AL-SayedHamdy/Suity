#importing_the_librabries
import os,cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import joblib as joblib
from keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

#Importing the data
data_dir = os.path.join('F:/Work/Projects/Graduation project/Clothes classifier/Data/Women')

categories = ['dresses_casual_women_clothes', 'evening_dresses_women_clothes',
              'jackets_top_sport_women_clothes','jens_bottom_casual_women_clothes',
              'jumpsuits_casual_women_clothes', 'pants_bottom_formal_women_clothes',
              'shorts_bottom_sport_women_clothes', 'skirts_bottom_casual_women_clothes',
              'skirts_bottom_formal_women_clothes', 'suits_bottom_formal_women_clothes',
              'suits_sport_women_clothes', 't_shirts_top_sport_women_clothes',
              'tops_casual_women_clothes', 'tops_top_formal_women_clothes', 'trousers_bottom_sport_women_clothes']

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
X = []
y = []

for features, labels in training_data:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1, img_size, img_size, 1)
y = np.array(y)

#Noramlize the data
X = X/255.0

#create_the_validation_dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=0)

#create_the_test_and_final_training_datasets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.78, random_state=0)

#model_building
model = Sequential()

#the_layers
model.add(Conv2D(32, (2, 2), activation='relu', padding = 'same', input_shape=(img_size, img_size, 1)))
model.add(Conv2D(32, (2, 2), activation='relu', padding = 'same'))


model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))

model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(15, activation='softmax'))
    
#compile_the_model
model.compile(optimizer='Nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

'''
#Data augmentation(needs a GPU)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=100,)

validation_generator = test_datagen.flow(
    X_val,
    y_val,
    batch_size=100,)

model.fit_generator(
    train_generator,
    steps_per_epoch=len(X_train) // 100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(X_val) // 100)
'''

history = model.fit(X_train, y_train, epochs = 10, batch_size = 100, validation_data=(X_val, y_val))

#The_visualisation
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()

'''
# save model
model.save('women_classifier-mod.h5')
'''

#test_the_model
y_pred = model.predict(X_test)

#convert_it_into_classes
y_classes = [np.argmax(element) for element in y_pred]

#testing_the_accuracy
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

score, acc = model.evaluate(X_test, y_test, batch_size=(100))

def plot_sample(x, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(x[index], cmap = 'gray')
    plt.xlabel(categories[y[index]])

plot_sample(X_test, y_test,26)