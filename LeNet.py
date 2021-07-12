import torch
from torch import nn
from train_nn_single_GPU import train_nn
from utility import load_data_fashion_mnist, try_gpu



class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

# define the LeNet
net = torch.nn.Sequential(
                          Reshape(), # accomodate the fashion-MNIST format
                          nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), 
                          nn.Sigmoid(),
                          nn.AvgPool2d(kernel_size=2, stride=2),
                          nn.Conv2d(6, 16, kernel_size=5), 
                          nn.Sigmoid(),
                          nn.AvgPool2d(kernel_size=2, stride=2), 
                          nn.Flatten(),
                          nn.Linear(16 * 5 * 5, 120), 
                          nn.Sigmoid(),
                          nn.Linear(120, 84), 
                          nn.Sigmoid(), 
                          nn.Linear(84, 10))


if __name__ == '__main__':

    train_iter, test_iter = load_data_fashion_mnist(batch_size=256)
    train_nn(net, train_iter, test_iter, num_epochs=10, lr=0.9, device=try_gpu())
