#**Behavioral Cloning** 
---

**Behavioral Cloning Project**

[//]: # (Image References)

[image1]: center_norm.jpg "Normal Image"
[image2]: center_flip.jpg "Flipped Image"
[image3]: recovery.gif "Recovery Image"

[image4]: straight.gif "Straight Image"
[image5]: figure_1.png "MSE"

---

####1. Included files

Project includes the following files:  
  
  * model.py containing the script to create and train the model
  * drive.py for driving the car in autonomous mode
  * model.h5 containing a trained convolution neural network 
  * README.md summarizing the results

####2. Code directions
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Included reusable code

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network modeled after the Nvidia network from their [deep learning paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf "Nvidia paper"). It starts with 5x5 filter sizes, then moves to 3x3 and depths are between 24 and 64 (model.py lines 164-176) 

The convolutional layers in the model include RELU activations to introduce nonlinearity (code line 164-168), and the data is normalized in the model using a Keras lambda layer (code line 163). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 171-175). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 158-159). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 178).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and two counter-clockwise laps.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach


My first step in deriving the model architecture was to use a convolution neural network model similar to the Nvidia model.  
I thought this model might be appropriate because of the numerous convolutional layers insuring detailed feature extraction.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. The picture below shows the final mean squared error for each set:

![alt text][image5]


To combat the overfitting, I modified the model to include L2 regularization and dropout layers after the fully-connected layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track on the first turn before the bridge. To improve the driving behavior in these cases, I drastically modified the augmenting procedures which provided the most substantial improvement.  
  
The Augmenting procedure is explained in detail in a further section.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 163-164) consisted of a convolution neural network with the following layers and layer sizes:  


| Layer         		|     Description	      | 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image | 
| Convolution 5x5   | 2x2 stride, valid padding, ReLU, depth 24|
| Convolution 5x5   | 2x2 stride, valid padding, ReLU, depth 36 |
| Convolution 5x5   | 2x2 stride, valid padding, ReLU, depth 48 |
| Convolution 3x3   | 2x2 stride, same padding, ReLU, depth 64 |
| Convolution 3x3   | 2x2 stride, valid padding, ReLU, depth 64 |
| Flatten           | 						      |
| Fully Connected	| outputs 100			      |
| Dropout           | probability	 0.5         |
| Fully Connected   | outputs 50              |
| Dropout				| probability	 0.5         |
| Fully Connected   | outputs 10              | 
| Dropout				| probability	 0.5         |
| Fully Connected   | outputs 1               |   


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track, one using center lane driving. Here is an example image of center lane driving:

![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer to the center if it was off to the side. This animation shows what a recovery looks like starting from the right :

![alt text][image3]


To augment the data set, I started by randomly selected one of the three different camera images; left, right, or center. Then I altered the brightness and shifted each image randomly along with altering the steering angle. Lastly, I also flipped images and angles in order to  decrease the bias towards any one angle since the track primarily consists of left turns. For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image2]


After the collection process, I had about 30k data points. I then preprocessed this data by cropping out the irrelevant parts of the image (car hood, upper horizon).


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the validation accruacy increasing to roughly 97% and stagnating around that point. I used an adam optimizer so that manually training the learning rate wasn't necessary.
