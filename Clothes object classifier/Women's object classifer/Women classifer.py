#Libraries
import os,cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

#Importing the data
data_dir = os.path.join('..//data directory//..')

categories = ['Dress', 'Foot wear', 'Hijab', 'Jacket', 'Jumpsuit', 'Shorts', 'Skirt', 'Sweatshirt', 'Top', 'Trousers']

#Show with gray scale
for category in categories:
    path = os.path.join(data_dir, category)
    for img in os.listdir(path):
        img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_arr, cmap = 'gray')
        plt.show()
        break
    break

batch_size = 32
img_size = 50
epochs = 15

#Show after the resize
new_arr = cv2.resize(img_arr, (img_size, img_size))
plt.imshow(new_arr, cmap = 'gray')
plt.show()

#Creat training data
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

#Suffle
random.shuffle(training_data)

#Test the shuffle
for sample in training_data[:20]:
    print(sample[1])


#Features and lables split
X = []
y = []

for features, labels in training_data:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1, img_size, img_size, 1)
y = np.array(y)

#Normalize the data
X = X/255.0

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

#Data augmentation
datagen = ImageDataGenerator(rotation_range=90,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             vertical_flip=True,
                             horizontal_flip=True)
datagen.fit(X_train)

#CNN model building
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (4, 4), activation='relu'))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))
    
#Compile
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Check point and early stopping
checkpoint = ModelCheckpoint("women_classifier.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

#Fitting the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

#Visualisation
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'])
plt.show()

'''
#Model save
model.save('women_classifier.h5')
'''

'''
#Model load
model = keras.models.load_model('..//model position//..')
'''

#Test the model
y_pred = model.predict(X_test)

y_classes = [np.argmax(element) for element in y_pred]

#The accuracy and loss
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

#Show sample
def plot_sample(x, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(x[index], cmap = 'gray')
    plt.xlabel(categories[y[index]])

plot_sample(X_test, y_test,14)
