# MNIST-neural-network-example
This is an example code that explains how to create a PyTorch neural network for MNIST digit recognition data

The code includes the following features:
* converting CSV data to suitable PyTorch tensors
* defining a two-layer neural network
* using confusion matrix as a metrics
* interpreting the weights of the neural network using 2D plots
* monitoring the learning process with tqdm progress bars

The resulting confusion matrix may look similar to:
```
[[4080    0   13    6    2   15   14    5    7   10]
 [   0 4616   14    8    6    7    3   13   22    5]
 [   6   11 4042   34    8    4   13   24   25    5]
 [   0    7   23 4168    3   49    0   15   44   32]
 [   3    5   19    4 3957    3    9   13   10   48]
 [   5    5    3   49    3 3655   17    2   20   16]
 [  18    3    7    7   14   21 4068    1    5    4]
 [   4    8   20   17   12    4    0 4273    5   61]
 [  10   24   27   38   10   18   10   10 3898   21]
 [   6    5    9   20   57   19    3   45   27 3986]]
```

This code is expected to run around 5 minutes on common hardware, and the resulting loss function value is close to 12.0.
