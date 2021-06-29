import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

"""
Implement the MLP from scratch, used to solve for the fashion-mnist classification problem
"""

def load_data_fashion_mnist(batch_size):

    mnist_train, mnist_test = keras.datasets.fashion_mnist.load_data()
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. And cast label to int32
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    return tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(batch_size).shuffle(len(mnist_train[0])), \
          tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(batch_size)


class Updater():
    """
    For updating parameters using minibatch stochastic gradient descent.
    """

    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    @staticmethod
    # Minibatch stochastic gradient descent.
    def sgd(params, grads, lr, batch_size):
      for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / batch_size)

    def __call__(self, batch_size, grads):
      self.sgd(self.params, grads, self.lr, batch_size)


def relu(X):
    return tf.math.maximum(X, 0)

# define a model
def net(X):
    X = tf.reshape(X, (-1, num_inputs)) #reshape each two-dimensional image into a flat vector of length num_inputs
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2

# define the loss function
def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(y, y_hat, from_logits=True) # recommend to set it as True for numerically stability. Then we should remove the last layer's softmax funtion


def accuracy(y_hat, y):
    """
    Compute the number of correct predictions.
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = tf.argmax(y_hat, axis=1)
    cmp = tf.cast(y_hat, y.dtype) == y
    return float(tf.reduce_sum(tf.cast(cmp, y.dtype)))


def evaluate_accuracy(net, data_iter):
    """
    Compute the accuracy for a model on a dataset.
    """
    metric = [0.0]*2 # No. of correct predictions, no. of predictions
    for X, y in data_iter:
      addon = accuracy(net(X), y), tf.size(y).numpy()
      metric = [a+b for a, b in zip(metric, addon)]
    return metric[0] / metric[1]

def train_epoch(net, train_iter, loss, updater):

    for X, y in train_iter:
        metric = [0.0] * 3 # Sum of training loss, sum of training accuracy, no. of examples
        # Compute gradients and update parameters
        with tf.GradientTape() as tape:
            y_hat = net(X)
            l = loss(y_hat, y)
            updater(X.shape[0], tape.gradient(l, updater.params))

        l_sum = tf.reduce_sum(l)
        addon = l_sum.numpy(), accuracy(y_hat, y), tf.size(y).numpy()
        metric = [a + b for a, b in zip(metric, addon)]

    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print('epoch {0}, train loss {1}, train acc {2}, test acc {3}'.format(epoch, train_metrics[0],
                                                                              train_metrics[1], test_acc))

def get_fashion_mnist_labels(labels):
    """
    Return text labels for the Fashion-MNIST dataset.
    """
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """
    Plot a list of images.
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.numpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


if __name__ == '__main__':

    # initializing parameters
    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    W1 = tf.Variable(tf.random.normal(shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
    b1 = tf.Variable(tf.zeros(num_hiddens))
    W2 = tf.Variable(tf.random.normal(shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
    b2 = tf.Variable(tf.random.normal([num_outputs], stddev=.01))

    params = [W1, b1, W2, b2]

    num_epochs = 10
    lr = 0.1 # learing rate
    updater = Updater([W1, W2, b1, b2], lr)

    # load data
    train_iter, test_iter = load_data_fashion_mnist(batch_size=256)
    # start to train the MLP model
    train(net, train_iter, test_iter, loss, num_epochs, updater)

    # load testing iter
    for X, y in test_iter:
        break

    # show the first 8 images in the testing data
    n = 8
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(tf.argmax(net(X), axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(tf.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n]);
