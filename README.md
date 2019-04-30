**Questions**

1. You are provided with a dataset of images which contains scatter plots and correlations corresponding to each scatter plot. Given these images, predict the correlation between x and y values. To get acquainted with how it works, you can play the game here first: 
[http://guessthecorrelation.com/](http://guessthecorrelation.com/)

    Follow the steps below to download the dataset and train a model.
    (1) Downloading the dataset
    * Download the images from this [link](https://drive.google.com/file/d/1kNgjfb3FF4pnGO__wy0hCgekKgfv5VMa/view?usp=sharing)
    * Download the correlation values from this [link](https://drive.google.com/file/d/1iUuhI78_8SW9MC6QB9wQeo0kjuAbk6AD/view?usp=sharing) 
    
    (2) The folder contains 150,000 images.      Separate the images into training and test sets.
    * How many images will you use for training?
    * How many images will you use for testing?  Why?
    
    (3) Create a convolutional neural network and train it using the dataset created in step 2. The model should not have more than two million parameters. You don't need to train the model for a long time, just a few iterations and to allow the loss to converge.
    * How did you calculate the number of parameters?
    * Which loss function did you use?
    * How many epochs was the model trained?
    * What was the loss before and after training?
   

2. Suppose you are tasked with creating a feature to enhance dark/bright image with a model as small as possible. That is, brighten too-dark images and darken too-bright images. Please provide a written response in English to the following questions concerning the above scenario:
    
    (1) Outline your plan to solve this problem, if you are given: (i) one week, (ii) one month, (iii) 6 months. In each case, specify what data you would need, and it is ok to suggest different options with pros/cons.

    (2) How to evaluate the model? (measure the success of this project)

**Usage**

1. [training_resnet.py](https://github.com/EileenHsieh/PicCollage_take_home_quiz/blob/master/scripts/training_resnet.py) is main script for model in [resnet18.py](https://github.com/EileenHsieh/PicCollage_take_home_quiz/blob/master/scripts/resnet.py) 
2. [training.py](https://github.com/EileenHsieh/PicCollage_take_home_quiz/blob/master/scripts/training.py) is main script for model in [vanilla_CNN.py](https://github.com/EileenHsieh/PicCollage_take_home_quiz/blob/master/scripts/vanilla_CNN.py) 
3. [get_sample_data.py](https://github.com/EileenHsieh/PicCollage_take_home_quiz/blob/master/scripts/get_sample_data.py) is for separating training set, vaildation set and test set.
4. [DataTool.py](https://github.com/EileenHsieh/PicCollage_take_home_quiz/blob/master/scripts/DataTool.py) defines the image preprocessing functions.
5. [playOnFireFox.py](https://github.com/EileenHsieh/PicCollage_take_home_quiz/blob/master/scripts/playOnFirefox.py) enables people to automatically play the game [online](http://guessthecorrelation.com/).

