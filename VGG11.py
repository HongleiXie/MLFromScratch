from torch import nn
from train_nn_single_GPU import train_nn
from utility import try_gpu, load_data_fashion_mnist


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1

    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    net = nn.Sequential(*conv_blks, nn.Flatten(), 
                        # 3 FC layers
                        nn.Linear(out_channels * 7 * 7, 4096), 
                        nn.ReLU(),
                        nn.Dropout(0.5), 
                        nn.Linear(4096, 4096), 
                        nn.ReLU(),
                        nn.Dropout(0.5), 
                        nn.Linear(4096, 10)
                        )
    return net


if __name__ == '__main__':

    # VGG-11: 8 Convs + 3 FC
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = vgg(conv_arch)
    lr, num_epochs, batch_size = 0.05, 1, 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size, resize=224)
    train_nn(net, train_iter, test_iter, num_epochs, lr, try_gpu())
