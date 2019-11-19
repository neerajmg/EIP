1st DNN Accuracy:
print(score)
[0.03734615365753525, 0.9916]


#Assignment Topics:

### Convolution

Convolution in neural nets is the operation of multiplying an input matrix such as an image with a kernel(another matrix) which is smaller in size than the input image. The multiplication is done one-on-one and the kernel is slided over the input to cover it entirely. It is done to extract important details from the input specific to the task 

![alt](https://cdn-images-1.medium.com/max/1600/1*Fw-ehcNBR9byHtho-Rxbtw.gif)

### Filters/Kernels

Filter/Kernel in a neural network refers to a matrix which is multiplied to input to obtain specific information. To the input a bunch of filters are applied each performing a different action and the information is passed on to the next layer. In older computer vision days the kernels are hand prepared but in neural networks backpropagation does the work for us.

![alt](https://cdn-images-1.medium.com/max/1600/1*_34EtrgYk6cQxlJ2br51HQ.gif)


### Epochs

To train a neural network inputs are sent in batches. When all the inputs from different batches sent combined equals total number of inputs in dataset it is considered an epoch.  A neural network is trained for a certain number of epochs until no improvement in accuracy is observed or until the network doesn't overfit.

![alt](https://kheangseng.files.wordpress.com/2010/09/how-to-choose-epochs1.jpg)

### 1x1 Convolution

In neural networks 1x1 convolutions are generally used to reduce the dimensions of input in the filter dimension. As the depth of the neural network is increased in general the filter dimension is increased which leads to a lot of computation. To save this effort 1x1 convolutions are used. They were extensively used in Google's inception architecture

![alt](https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed_small.gif)

### 3x3 Convolution

3x3 Convolution is the standard convolution used in the latest neural network architectures. All major hardware are made efficient to run 3x3 convolutions hence using 3x3 convolutions provides an advantage. Any higher dimension convolution can be represented by multiple 3x3 convolution such as receptive field of 1 5x5 is same as 2 3x3 stacked together. 

![alt](https://mlnotebook.github.io/img/CNN/convSobel.gif)

### Feature Maps
A feature map is a mapping of the location of certain feature in an image.  
![Feature Maps](https://raw.githubusercontent.com/sin2akshay/External-Internship-Program-2.0-Machine-Learning-for-Deep-Neural-Networks/master/Session%201/_files/Feature%20Maps.JPG)  
In easier words, a feature map is produced when convolution is applied to the input data using a convolution filter. Suppose we have an input and 3x3 filter/kernel as shown below:  
![Input matrix and kernel matrix](https://cdn-images-1.medium.com/max/800/1*cTEp-IvCCUYPTT0QpE3Gjg@2x.png)  
We do the convolution using this filter and moving it over the input. The result goes into the feature map (3x3 matrix on right side) as shown below:  
![Animation | Feature Maps](https://cdn-images-1.medium.com/max/800/1*VVvdh-BUKFh2pwDD0kPeRA@2x.gif)  
We perform multiple convolutions on the input image using different filters to generate different feature maps. These feature maps are stacked together to become the output convolution layer.


### Activation Function
In an Artificial Neural Network, the activation function of a neuron defines the output of that neuron given a set of inputs.

To understand this, we can say that activation function is biologically inspired by the activities in our brain, where in a similar fashion different neurons are fired/activated for different stimuli.

![Neurons being activated by different stimuli](https://raw.githubusercontent.com/sin2akshay/External-Internship-Program-2.0-Machine-Learning-for-Deep-Neural-Networks/master/Session%201/_files/neuron.jpg)  

For each different stimuli certain neurons fire. So within our brain, neurons are either firing or they are not. We can interpret this as a binary 'on' or 'off', 0 or 1. For example, In sigmoid activation function the neuron can be between 0 and 1. So the more closer to 1, the more activated that neuron is and the closer to 0, the less activated that neuron is.  


### Receptive Field

A receptive field in a CNN is a part of the input image, i.e. pixel grid that is visible to a feature/kernel during the sliding/convolution operation. This is equal to the kernel/filter size. This receptive field increases linearly as we stack more convolutional layers.

Each unit in a hidden layer is only connected to a small number of units in the previous layer. For instance a node in the first hidden layer will only be connected to a small patch of region of the input image which is the receptive field of that layer.
