# SVRG-Pytorch
Implementation of an efficient variant of SVRG that relies on mini-batching implemented in Pytorch

## Implementation Details

This implementation of SVRG combines practical changes introduced in ['Stop Wasting My Gradients: Practical SVRG'](https://arxiv.org/abs/1511.01942) and ['Mini-Batch Semi-Stochastic Gradient Descent in the Proximal Setting'](https://arxiv.org/abs/1504.04407).

From the first paper, instead of computing the full gradient, they use a large batch size to estimate the full gradient. From the second paper, instead of updating the parameters by using gradients from a single training example, they use mini-batching.

Feel free to make any changes that improve the efficiency of the optimizer, or point out any mistakes.

## Requirements

pytorch version 1.1.

autograd (if you want to run the MNIST Example. I had trouble using the built in pytorch dataloader).

## Example
In the example file, we train a feedforward neural network to classify [MNIST](http://yann.lecun.com/exdb/mnist/) digits. 
