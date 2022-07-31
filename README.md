# Sauty
Clothes recommender system

## General info
Suity is a clothes recommender system that depends on the clothes you will give him as input and every day, he will recommend an outfit that will fit you, depending on the weather and the style you like to wear.
And this is the Artificial intelligence part of this application.

## Technologies
Project is created with:
* Convolution neural network
* TensorFlow version: 2.1.0
* Keras version: 2.2.4-tf
* scikit-learn version: '1.0.2'
* OpenCV
	
## CNN Structures for the deep learning classification model

----------------------------------------------------------------
	Layer (type)                 Output Shape              Param #   

	conv2d_1 (Conv2D)            (None, 48, 48, 32)        320       

	max_pooling2d_1 (MaxPooling2 (None, 24, 24, 32)        0         

	dropout_1 (Dropout)          (None, 24, 24, 32)        0         

	conv2d_2 (Conv2D)            (None, 22, 22, 64)        18496     

	max_pooling2d_2 (MaxPooling2 (None, 11, 11, 64)        0         

	dropout_2 (Dropout)          (None, 11, 11, 64)        0         

	conv2d_3 (Conv2D)            (None, 8, 8, 128)         131200    

	dropout_3 (Dropout)          (None, 8, 8, 128)         0         

	flatten_1 (Flatten)          (None, 8192)              0         

	dense_1 (Dense)              (None, 512)               4194816   

	dropout_4 (Dropout)          (None, 512)               0         

	dense_2 (Dense)              (None, 7)                 3591      

Total params: 4,348,423
Trainable params: 4,348,423
Non-trainable params: 0

    
## Output for men

![Sample Output](https://github.com/AL-SayedHamdy/Men-and-women-clothes-classification/blob/main/Images/Men%20classefier.png)

## Output for women

![Sample Output](https://github.com/AL-SayedHamdy/Men-and-women-clothes-classification/blob/main/Images/Women%20classifier.png)
