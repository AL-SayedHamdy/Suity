#importing_the_librabries
import os,cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers.experimental import preprocessing
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#importing_the_data
data_dir = os.path.join('../data directory/..')

categories = ['jackets_top_casual_men_clothes', 'jackets_top_formal_men_clothes',
              'jackets_top_sport_men_clothes','shirts_top_casual_men_clothes',
              'shirts_top_formal_men_clothes', 'shorts_bottom_casual_men_clothes',
              'shorts_bottom_sport_men_clothes', 'suits_sport_men_clothes',
              'sweatshirt_top_casual_men_clothes', 't_shirts_top_casual_men Clothes',
              't_shirts_top_casual_men Clothes', 'trousers_bottom_casual_men clothes',
              'trousers_bottom_formal_men_clothes', 'trousers_bottom_sport_men_clothes']

#show_with_the_garyscale
for category in categories:
    path = os.path.join(data_dir, category)
    for img in os.listdir(path):
        img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_arr, cmap = 'gray')
        plt.show()
        break
    break

#image_size
img_size = 50
new_arr = cv2.resize(img_arr, (img_size, img_size))
plt.imshow(new_arr, cmap = 'gray')
plt.show()

#training_data
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

#suffeling_the_data
random.shuffle(training_data)

#test_shuffeling
for sample in training_data[:20]:
    print(sample[1])


#features_and_labels
X = []
y = []

for features, labels in training_data:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1, img_size, img_size, 1)
y = np.array(y)

#noramlize_the_data
X = X/255.0

#create_the_validation_dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=0)

#create_the_test_and_final_training_datasets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.78, random_state=0)

    
#model_building
model = Sequential()

#the_layers
model.add(Conv2D(64, (4, 4), activation='relu', kernel_initializer='he_uniform', input_shape=(img_size, img_size, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(500, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.25))

model.add(Dense(14, activation='softmax'))
    
#compile_the_model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs = 10, batch_size=65, validation_data=(X_val, y_val))

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
model.save('men_classifier_mod2.h5')
'''
#test_the_model
y_pred = model.predict(X_test)

#convert_it_into_classes
y_classes = [np.argmax(element) for element in y_pred]

#testing_the_accuracy
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

score, acc = model.evaluate(X_test, y_test, batch_size=(65))

def plot_sample(x, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(x[index], cmap = 'gray')
    plt.xlabel(categories[y[index]])

plot_sample(X_test, y_test,77)
