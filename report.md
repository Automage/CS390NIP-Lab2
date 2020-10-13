# Lab 2 Report
Pranav Nair
nair51@purdue.edu

Repo: https://github.com/Automage/CS390NIP-Lab2

## Resources
https://keras.io/api/layers/regularization_layers/dropout
https://www.cs.toronto.edu/~kriz/cifar.html
http://d2l.ai/chapter_convolutional-modern/vgg.html

## Questions
### How is a CNN superior to standard ANNs for image processing?
CNNs are superior to standard ANNs due to their ability to extract data more relevant to that of an image. By considering the relationship between each pixel's neighbors (via filters and convolutions), the data provided to the fully connected layer is more meaningful compared to a one dimensional vector of pixels, leading networks to 'learn' better.

### Why do we sometimes use pooling in CNNs?
Pooling is used to reduce the dimensionality of the image tensor from convolution layers by removing unwanted data. This allows the processing of further layers to be less demanding on the processor.

### Why do you think the cifar datasets are harder than mnist?
Cifar datasets have a larger input compared to the MINST dataset – the dimensions of the images are larger as well as containing a color component, adding an extra dimension to the tensor. Due to this, networks are unable to classify the more complex cifar dataset as well as the simpler MINST dataset, which contain 2 dimensional tensors as input.

## CNN hyperparameters and architecture
There were several optimizations made to the standard CNN discussed in lecture in order to improve accuracy. 

The first was to implement the concept of dropout, which would help the probability of the network overfitting the training data. Two dropout layers were added, one before the first fully connected layer, and before the second fully connected layer.

A batch size of 64 was also used in order to increase the effectiveness of the training process of the network. 

The final optimization made was to update the architecture of the network. Specifically, 3 VGG blocks were added, each containing two convolution layers and one max pooling layer. The filter size was increased gradually between the three layers, 32, 64 and 128 respectively. 

### Hyperparameters
- loss type: categorical crossentropy
- convolution layers with size 32, 64 and 128 respectively
- max pooling size of (2,2)
- epochs: 10, batch-size: 64
- 0.2 dropout rate
- kernel size was (3,3)
- Two fully connected layers with 10 and 512 nodes, respectively.s
- ReLu activation was used in the convolution and fully connected layers, expect the final dense layer which used softmax.

## Outputs

### ANN

#### MINST-d
F - scores:  [0.95760599 0.97762176 0.93020986 0.9261811  0.93138244 0.9224537
 0.94736842 0.93473896 0.90072389 0.90732654]

#### MINST-f
#### Cifar-10
#### Cifar-100-c
#### Cifar-100-f

### CNN

#### MINST-d
#### MINST-f
#### Cifar-10
#### Cifar-100-c
#### Cifar-100-f
