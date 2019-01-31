# Install
First install pytorch (https://pytorch.org/) and then tensorboardX (https://github.com/lanpa/tensorboardX).

#  gans: Generative Adversarial Networks
Multiple Generative Adversarial Networks (GANs) implemented in PyTorch and Tensorflow.

[Check out this blog post](https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f) for an introduction to Generative Networks. 


<img src=".images/dcgan_mnist.gif" width="275"> <img src=".images/dcgan_cifar.gif" width="275">

## Vanilla GANs
Vanilla GANs found in this project were developed based on the original paper [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) by Goodfellow et al.

These are trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), and learn to create hand-written digit images using a 1-Dimensional vector representation for 2D input images. Run these jupyter notebooks on our machine. 
- [PyTorch Notebook](https://github.com/Shintaro-Miyazaki/gans/blob/master/1.%20Vanilla%20GAN%20PyTorch.ipynb)
- [TensorFlow Notebook](https://github.com/Shintaro-Miyazaki/gans/blob/master/1.%20Vanilla%20GAN%20TensorFlow.ipynb)

<img src=".images/vanilla_mnist_pt_raw.png" width="300"> <img src=".images/vanilla_mnist_pt.png" width="300">

__MNIST-like generated images before & after training.__


## DCGANs
Deep Convolutional Generative Adversarial Networks (DCGANs) in this repository were developed based on the original paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) by Radford et al.

These are trained on the CIFAR10 and the MNIST datasets. They use 3 dimensional representations for images (length x height x colors) directly for training.

- [DC-GAN PyTorch MNIST Notebook](https://github.com/Shintaro-Miyazaki/gans/blob/master/2.%20DC-GAN%20PyTorch-MNIST.ipynb)
- [DC-GAN PyTorch CIFAR Notebook](https://github.com/Shintaro-Miyazaki/gans/blob/master/2.%20DC-GAN%20PyTorch.ipynb)
- [DC-GAN Tensorflow CIFAR Notebook](https://github.com/Shintaro-Miyazaki/gans/blob/master/2.%20DC-GAN%20TensorFlow.ipynb)

<img src=".images/dcgan_cifar_pt_raw.png" width="300"> <img src=".images/dcgan_cifar_pt.png" width="300">

__CIFAR-like generated images before & after training.__
