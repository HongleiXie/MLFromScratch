import torchvision
from torch.utils import data
from torchvision import transforms
import torch
import numpy as np
import time


def load_data_fashion_mnist(batch_size, resize=None):
    
    """
    Download the Fashion-MNIST dataset and then load it into memory.
    """

    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root="~/Downloads/data",
                                                    train=True,
                                                    transform=trans,
                                                    download=True)

    mnist_test = torchvision.datasets.FashionMNIST(root="~/Downloads/data",
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=2),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=2))

class Accumulator:
    """
    For accumulating sums over `n` variables.
    """

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

def try_gpu(i=0):

    """
    Return gpu(i) if exists, otherwise return cpu().
    """

    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)

    cmp = y_hat.type(y.dtype) == y
    cmp = cmp.type(y.dtype)
    return float(torch.sum(cmp))