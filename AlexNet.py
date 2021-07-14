from torch import nn
from train_nn_single_GPU import train_nn
from utility import try_gpu, load_data_fashion_mnist


net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), 
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), 
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), 
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), 
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), 
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2), 
    nn.Flatten(),
    nn.Linear(6400, 4096), 
    nn.ReLU(), 
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), 
    nn.ReLU(), 
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
    )


if __name__ == '__main__':

    lr, num_epochs, batch_size = 0.01, 10, 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size, resize=224)
    train_nn(net, train_iter, test_iter, num_epochs, lr, try_gpu())