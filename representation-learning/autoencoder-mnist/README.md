# Autoencoder for MNIST dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ewV8rOu_1dEKn6EYoYFSwQn4kseE-5Sj)

An **autoencoder** is a neural network used for *unsupervised learning* capable of creating a representation of the input data in an efficient way.

In an autoencoder the output data are identical to the input data. The goal of an autoencoder is to represent the structure of the data.
They observe the input data, process an efficient representation of the same, producing outputs similar to the input data. 
The network compresses the incoming data into a latent space and reconstructs it from this same space.

In this project I built two autoencoders. The first is a simple autoencoder.
The aim of the second autoencoder is to remove the noise from the images (*image denoising*).

## Dataset
The dataset is provided by [Tensorflow](https://www.tensorflow.org/datasets/catalog/mnist).