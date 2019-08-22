import csv
import numpy as np
import pickle
import cv2
from sklearn.model_selection import train_test_split
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten,Activation,Cropping2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
import math
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import load_model


### to avoid my GPU specific error. ###
import tensorflow as tf
from tensorflow.python.keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
###########################################3


### Define generator ###


# split data into image and steer angle
# Generator which is called per batch

from sklearn.utils import shuffle
def driving_data_generator(input_data,batch_size): # this input does not mean network input,just generator's input.
    data_length = len(input_data)
    while 1:
        shuffle(input_data)
        for offset in range(0,data_length,batch_size):
            batch_data = input_data[offset:offset+batch_size]
            # hold network input and ouput data
            images = []
            angles = []
            for data in batch_data:

                for is_flip in range(2):
                    if is_flip == 0:
                        for direction in range(3):
                            name = data[direction]
                            image = cv2.imread(name)
                            images.append(image)
                            if direction == 0:
                                angle = float(data[3])
                            elif direction == 1:
                                angle = float(data[3])+0.1
                            else:
                                angle = float(data[3])-0.1
                            angles.append(angle)
                    elif np.abs(float(data[3])) > 0.05:
                        for direction in range(3):
                            name = data[direction]
                            image = cv2.imread(name)
                            image = np.fliplr(image)
                            images.append(image)
                            if direction == 0:
                                angle = (float(data[3]))*(-1)
                            elif direction == 1:
                                angle = (float(data[3])+0.1)*(-1)
                            else:
                                angle = (float(data[3])-0.1)*(-1)
                            angles.append(angle)

            #endfor
            yield np.array(images),np.array(angles)


########################


### Load data-set ###

driving_data_center = []
# Load driving data in center lane
with open("/home/nishi/udacity/sim_data_center/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for data_line in reader:
        driving_data_center.append(data_line)
        
    #endfor
#endwith
"""
driving_data_additional=[]
with open("/home/nishi/udacity/sim_data_additional/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for data_line in reader:
        driving_data_additional.append(data_line)
        
    #endfor
#endwith
"""

driving_data_left = []
# Load driving data in left lane
with open("/home/nishi/udacity/sim_data_recovery_left/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for data_line in reader:
        driving_data_left.append(data_line)
        
    #endfor
#endwith

with open("/home/nishi/udacity/sim_data_recover_left2/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for data_line in reader:
        driving_data_left.append(data_line)
        
    #endfor
#endwith

driving_data_right = []
# Load driving data in right lane
with open("/home/nishi/udacity/sim_data_recovery_right/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for data_line in reader:
        driving_data_right.append(data_line)
        
    #endfor
#endwith
"""
with open("/home/nishi/udacity/sim_data_recover_right2/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for data_line in reader:
        driving_data_right.append(data_line)
        
    #endfor
#endwith
"""

#################################################################

### Remove useless data ###


# Remove noisy driving data in left lane 
angle_data = np.array(driving_data_left)[:,3].astype("float")
noise_left = np.where(np.zeros_like(angle_data)>=angle_data)
driving_data_left=np.delete(driving_data_left,noise_left,axis=0)

# Remove noisy driving data in right lane 
angle_data = np.array(driving_data_right)[:,3].astype("float")
noise_right = np.where(np.zeros_like(angle_data)<=angle_data)
driving_data_right=np.delete(driving_data_right,noise_right,axis=0)

#################################################################





### define the model ###
# this model is based on NVIDIA's Network https://devblogs.nvidia.com/deep-learning-self-driving-cars/

### model configuration is here###
"""
# Set our batch size
batch_size=28
height = 70
width = 320
channel =3
# compile and train the model using the generator function
train_generator = driving_data_generator(train_data, batch_size=batch_size)
validation_generator = driving_data_generator(validation_data, batch_size=batch_size)
# I chose elu as activation funciton ,because the authors said that is best in this paper. (https://arxiv.org/pdf/1511.07289v5.pdf)
model = Sequential()
model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(height,width,channel)))
model.add(Conv2D(24,kernel_size=(5,5),strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(36,kernel_size=(5,5),strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(48,kernel_size=(5,5),strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(64,kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(64,kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Flatten())
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dense(50))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dense(10))
model.add(Dense(1))
model.summary()
model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator,
            steps_per_epoch=math.ceil(len(train_data)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_data)/batch_size), 
            epochs=10,callbacks=[EarlyStopping(patience=2)], verbose=1)

model.save('./model_20190820_without_dropout.h5')
print("finish!")
"""
#######################################################################


### Train with various parameter setting ###
"""
Introduce new parameter which express the ratio between main driving data(in center lane) and additional driving data(in side lane).
alpha = (the number of data driving in side lane / the number of data driving in center lane)
"""

alpha = np.arange(2,10,1)/10

# Set our batch size
batch_size=28

# change parameter value and train the model. 
for i in range(len(alpha)):
    print(len(driving_data_center))
    print(len(driving_data_center)-len(driving_data_left)/alpha[i])
    driving_data_center_modified=np.delete(driving_data_center,random.sample(list(np.arange(len(driving_data_center))), int(len(driving_data_center)-len(driving_data_left)/alpha[i])),axis=0)
    driving_data=np.concatenate((driving_data_center_modified,driving_data_left,driving_data_right),axis=0)
    # split data
    train_data, validation_data = train_test_split(driving_data, test_size=0.2) 

    
    # compile and train the model using the generator function
    train_generator = driving_data_generator(train_data, batch_size=batch_size)
    validation_generator = driving_data_generator(validation_data, batch_size=batch_size)
    # I chose elu as activation funciton ,because the authors said that is best in this paper. (https://arxiv.org/pdf/1511.07289v5.pdf)
    model = load_model('./model_20190820_without_dropout.h5')
    for j in range(len(model.layers)-2):
        model.layers[j].trainable=False
    #endfor

    model.summary()
    model.compile(loss='mse',optimizer='adam')
    model.fit_generator(train_generator,
                steps_per_epoch=math.ceil(len(train_data)/batch_size), 
                validation_data=validation_generator, 
                validation_steps=math.ceil(len(validation_data)/batch_size), 
                epochs=10,callbacks=[EarlyStopping(patience=2)], verbose=1)

    model.save('./model_alpha_{0}.h5'.format(alpha[i]*10))
    print("{0}/8 finish!".format(i+1))

#endfor

### my best model is ""model_alpha_2.0.h5" and this is renamed "model.h5" in my repository.



