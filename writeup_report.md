# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./center.jpg "Driving in center lane Image"
[image3]: ./left.jpg "Recovery Image"
[image4]: ./right.jpg "Recovery Image"
[image5]: ./flip.png "Flip Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the NVIDIA's CNN model introduced in the Udacity lesson. (model.py lines 171-197) 

The model includes ELU layers to introduce nonlinearity (code line 176 ,179, ...), and the data is normalized in the model using a Keras lambda layer (code line 18). 
I chose elu as activation funciton ,because the authors said that is best in this paper. (https://arxiv.org/pdf/1511.07289v5.pdf)

#### 2. Attempts to reduce overfitting in the model

The model contains Batch normalization layers which is said that it including work like dropout layer and avoid overfitting (model.py lines 175). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 230). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 199).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Nvidia's one. I thought this model might be appropriate because it has achieved self-driving.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model.
I introduced Batch normalization and dropout layer.(※I don't use dropout layer in last model)

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. In these case, I realize that steer angle value sometimes bacome odd as if the curve direction is opposit.To improve the driving behavior in these cases, I modify data distribution because my recovery driving data is inclined at that time.(Because of curve tendency of simulator cource)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to recover. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]

After the collection process, I had (center,left,right) = (10665,1472,1466) number of data points. 
Then, I removed useless data.
For examle, in recovery section, the data collected when car approached to lane line is not desired data.
When drive nearby left lane in recovery driving, firstly I approched a lane line and then I steered as I approched a line enough. So I delete the data which steering value is saved when I approached a lane line.

To augment the data set, I also flipped images and angles.
 For example, here is an image that has then been flipped:

![alt text][image5]

Besides, I use left and right camera with steer angle correction.

I then preprocessed this data by Lambda Layer, pixel value normalization, and clip image,top 70 pixel and bottom 20 pixel are removed.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary and I use EarlyStopping callback function so I didn't need to tune the number of epoch.  

Lastly I introduce new parameter which express the ratio of driving in center lane and recovery driving.(code in line 218)
This is because I think the ratio affects the driving tendency.

Best loss value does not mean bast model because data distribution is not same among all alpha values.
So I compare model by testing in simulation.