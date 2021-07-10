import torch
from torch import nn

"""
implement the 2D convolution layer without padding and stride = 1 and channel = 1

"""

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1)) # number of channels

    def corr2d(self, X, K):
      h, w = K.shape
      Y = torch.zeros(X.shape[0]-h+1, X.shape[1]-w+1) # result

      for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
          Y[i,j] = (X[i:i + h, j:j + w] * K).sum()
      return Y

    def forward(self, x):
        return self.corr2d(x, self.weight) + self.bias # broadcasting applied


if __name__ == '__main__':

    X = torch.tensor([
                  [0.0, 1.0, 2.0],
                  [3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0]
                  ])
    conv = Conv2D(kernel_size=(2,2))
    print(conv(X).data)
