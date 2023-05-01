# object_detection_in_video
Using Yolo algorithm

ml methods -sift,support vector machine,wireless john's object detection framework

Deep learning which is also called deep structured learning is a class of ml algo.Deep learning uses a multi layer approach to extract high-level feature that is provided to it

deep learning does not require any feature to be provided manually for classification instead it tries to transform its data into an abstract representation

deep learning algo-rcnn ,faster rcnn, yolo algo

cnn takes an input image assign importance to an object and able to differentiate between one object and other
cnn works by extracting features of a n object
cnn-convolutional neural networks are made up of neurons with learnable weights and biases. each neuron receives several inputs, take a weighted sum over them, pass it through an activation function and responds with an output.
cnn takes input as grayscale img the output layer is binary or multiclass labels and then we contain hidden layers which contains convolution layers, relu and pooling layers

rcnn-region based convolution network
in rcnn it works on dividing image into parts and then assign propability values to those parts and whichever part has highest probability its where we consider an object to be present

in yolo algo it focuses on the entire image as a whole and predicts the bounding boxes and then calculate the class probability to label the boxes

the family of yolo framework is very fast as compared to rcnn
 
the latest version of yolo is v3 which uses logistic classification which helps us in making multi layer classification and uses darknet53 as a extractor

darknet53 means 53 convolution layers which makes it help in predicting objects more accurately
