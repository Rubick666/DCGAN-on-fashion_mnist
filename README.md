DCGAN on Fashion MNIST Dataset

This repository contains an implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) using Keras to generate fashion images similar to those in the Fashion MNIST dataset.
Project Overview

Generative Adversarial Networks (GANs) are a class of deep learning models used for generating new data samples that resemble a given dataset. This project implements a DCGAN architecture consisting of a generator and a discriminator network trained adversarially:

    Generator: Takes random noise as input and generates fake fashion images.

    Discriminator: Tries to distinguish between real images from the Fashion MNIST dataset and fake images produced by the generator.

The goal is to train the generator to produce realistic fashion images that can fool the discriminator.
Dataset

    Fashion MNIST: A dataset of 28x28 grayscale images of 10 different clothing categories.

    The images are normalized to the range [-1, 1] for better GAN training stability.

Model Architecture

    The Generator uses dense layers, reshaping, upsampling, convolutional layers, and batch normalization to produce 28x28 grayscale images.

    The Discriminator uses convolutional layers with LeakyReLU activations, dropout, batch normalization, and a sigmoid output to classify images as real or fake.

    The Combined model stacks the generator and discriminator for training the generator to fool the discriminator.

Usage
Requirements

    Python 3.x

    Keras (with TensorFlow backend)

    numpy

    matplotlib

You can install the required packages with:

bash
pip install keras tensorflow numpy matplotlib

Running the project

    Clone the repository and navigate to the project directory.

    Run the Jupyter notebook fashion_mnis_DCGAN.ipynb or convert it to a Python script.

    The notebook trains the DCGAN model for a specified number of epochs.

    Generated images are saved periodically during training as gan_generated_image_epoch_<epoch>.png.

Training function parameters

    epochs: Number of training epochs.

    batch_size: Number of samples per training batch.

    save_interval: Interval (in epochs) at which generated images are saved.

Example to start training:

python
train(epochs=10000, batch_size=128, save_interval=1000)

Results

During training, the generator progressively learns to create more realistic images. Sample generated images at different epochs are saved in the project directory.
Project Structure

    fashion_mnis_DCGAN.ipynb : Main notebook containing the full implementation.

If you are able to run it for 10000 epoch, you will see magnificant result. An example of this is added to the repo
References

    Fashion MNIST dataset

    DCGAN paper

    Keras documentation for GANs
