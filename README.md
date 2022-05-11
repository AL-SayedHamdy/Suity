# Men's and women's clothes classification
It's a clothes category classifier.

## General info
In this project, I made a classifier that will be able to classify the men's and women's clothes to 14 categories for men's clothes and 15 for women

## Technologies
Project is created with:
* Convolution neural network
* TensorFlow version: 2.1.0
* Keras version: 2.2.4-tf
* scikit-learn version: '1.0.2'
* OpenCV
	
## CNN Structures for the men classifier (Building a model on my own)

----------------------------------------------------------------
        Layer (type)               Output Shape         Param 

    conv2d_1 (Conv2D)            (None, 47, 47, 64)        1088      

    max_pooling2d_1 (MaxPooling2 (None, 23, 23, 64)        0         

    flatten_1 (Flatten)          (None, 33856)             0         

    dense_1 (Dense)              (None, 500)               16928500  

    dropout_1 (Dropout)          (None, 500)               0         

    dense_2 (Dense)              (None, 14)                7014  
    
    
 
## CNN Structures for the women classifier (Building a model on my own)

----------------------------------------------------------------
        Layer (type)               Output Shape         Param 

    conv2d_1 (Conv2D)            (None, 49, 49, 32)        160       

    conv2d_2 (Conv2D)            (None, 48, 48, 32)        4128      

    max_pooling2d_1 (MaxPooling2 (None, 24, 24, 32)        0         

    dropout_1 (Dropout)          (None, 24, 24, 32)        0         

    conv2d_3 (Conv2D)            (None, 22, 22, 64)        18496     

    conv2d_4 (Conv2D)            (None, 20, 20, 64)        36928     

    max_pooling2d_2 (MaxPooling2 (None, 10, 10, 64)        0         

    dropout_2 (Dropout)          (None, 10, 10, 64)        0         

    flatten_1 (Flatten)          (None, 6400)              0         

    dense_1 (Dense)              (None, 512)               3277312   

    dropout_3 (Dropout)          (None, 512)               0         

    dense_2 (Dense)              (None, 15)                7695 
    
## Output for men

![Sample Output](https://github.com/AL-SayedHamdy/Men-and-women-clothes-classification/blob/main/Images/Men%20classefier.png)

## Output for women

![Sample Output](https://github.com/AL-SayedHamdy/Men-and-women-clothes-classification/blob/main/Images/Women%20classifier.png)
