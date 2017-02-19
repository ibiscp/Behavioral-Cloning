#Behavioral Cloning
The idea of this project is to clone the human behavior to learn how to steer the car in a simulated track. The only information given to the model is a front view of the vehicle and the output expected is the steering angle of the wheel.

In order to meet this goal, it is necessary to train a Convolutional Neural Network, so it learns what to do in each scenario given a dataset with a bunch of images and a steering angles related to each one.

The figure bellow presents an image of both tracks of the [simulator](###simulator-download):

<div style="text-align:center">
<img src="images/1st_track.png" style="width:100px;"/>
<img src="images/2nd_track.png" style="width:100px;"/>
</div>

##Resources
There are a few files needed to run the Behavioral Cloning project.

The simulator contains two tracks. Sample driving data for the first track is included bellow, which can optionally be used to help train the network. It is also possible to collect the data using the record button on the simulator.

* [sample data for track 1](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

###Simulator Download
* [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)
* [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)
* [Windows 32-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip)
* [Windows 64-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)

###Beta Simulators
* [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5894ee55_beta-simulator-linux/beta-simulator-linux.zip)
* [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5894ecbd_beta-simulator-mac/beta-simulator-mac.zip)
* [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5894ea69_beta-simulator-windows/beta-simulator-windows.zip)

## Dataset
The dataset, provided by Udacity, found in [this link](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip), contains the following data:

* Folder with 8.036 simulation images, showing the center, left and right camera view of the road, tantalizing 24.108 images
* File <em>driving\_log.csv</em> containing a list describing all the images with the following information
    * Center image path
    * Left image path
    * Right image path
    * Steering angle
    * Throttle
    * Brake
    * Speed

Bellow is an example of the images used to train the CNN, it is also shown how the steering angle is adjusted based on the image.
<br><br>
<div style="text-align:center"><img src="images/center_left_right.png"/></div>

## Histogram
The image bellow presents the histogram of the given dataset, here is possible to notice that the number of images with steering angle equal to zero is much more representative.
<br><br>
<div style="text-align:center"><img src="images/histogram_dataset.png"/></div>

In order to have a more balanced dataset, it is necessary to eliminate good part of the zero angle steering examples. It was decided to consider only 8% of the total number, and the result is presented bellow.

<div style="text-align:center"><img src="images/histogram_non_zeros.png"/></div>

## Image Augmentation
In order to improve the learning task and make it more robust, it is necessary to augment the dataset, so more data is artificially generated based only on the given ones.

The following augmentation is used in this project:

* Flip
* Change image brightness
* Rotate
* Translate
* Shadow
* Shear
* Cut

Examples of each transformation will be presented bellow.

###Flip
In order to have a balanced dataset, it is useful to flip each image randomly, also inverting the sign of the steering angle.
<br><br>
<div style="text-align:center"><img src="images/flip.png"/></div>

###Change image brightness
It is useful to change the image brightness in order to make the model learn how to generalize from a day to a rainy day or at night, for example. This can be achieved changing the V value of the converted image to HSV.
<br><br>
<div style="text-align:center"><img src="images/bright.png"/></div>

###Rotate
It is also possible to generate sloping angles, so the model learns how to generalize to these cases.
<br><br>
<div style="text-align:center"><img src="images/rotate.png"/></div>

### Translate
Translating the image randomly makes it possible to generate even more data in different positions of the road, adding a proportional factor of this translation to the steering angle.
<br><br>
<div style="text-align:center"><img src="images/translate.png"/></div>

### Shadow
Shading randomly an image makes it more robust to shadows on the track, such as a tree, wires or poles.
<br><br>
<div style="text-align:center"><img src="images/shadow.png"/></div>

### Shear
Shearing the image is also usefull, once it is possible to generate more data with the ones we already have, change the borders that the vehicle does not need to learn.
<br><br>
<div style="text-align:center"><img src="images/shear.png"/></div>

### Cut
In order to minimize the number of parameters of our CNN, it is possible to cut some unnecessary parts of the image, including the bottom, top and some few pixels on the sides.
<br><br>
<div style="text-align:center"><img src="images/cutted.png"/></div>

### Composed result
The image bellow shows an example of a composed treatment of an image.
<br><br>
<div style="text-align:center"><img src="images/composed.png"/></div>

## Neural Network Architecture
This project was tested using two diffenrent architectures, [CommaAI](https://github.com/commaai/research/blob/master/train_steering_model.py) and [NVIDIA](https://arxiv.org/pdf/1604.07316v1.pdf). Botch were trained using the same configuration (learning rate, optimizer, number of epochs, samples per epoch and augmentation) the only thing that was really changed was the model.

### Configuration
* Learning rate: 1e-4
* Optimizer: Adam
* Number of epochs: 10
* Samples per epoch: 20000
* Batch size: 50
* Validation split: 30%

### NVIDIA Architecture 


### CommaAI Architecture



## Results
Bellow is presented a video result running on the same track where the CNN was trained (Track 1). It was also tested on a track never seen before (Track 2) in order to prove that the model learns how to generalize to different tracks and conditions.

## Conclusion and next steps
The task of adjusting the parameters, in order to get a satisfactory result is really difficult. Besides defining the architecture parameters, various others factors influence on the result, such as augmentation and dataset balance.

For this task it is important to have a good computer in order to train the model faster. On my computer, with a NVIDIA GeForce GT 730M it takes about 20 minutes to train, what is a little bit frustating.