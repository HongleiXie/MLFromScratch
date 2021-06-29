import tensorflow as tf
import random


def synthetic_data(w, b, num_examples):
    """
    y = Xw + b + noise
    """
    X = tf.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = tf.reshape(y, (-1, 1))
    return X, y

def linreg(X, w, b):
    return tf.matmul(X, w) + b

def squared_loss(y_hat, y):
    # note: we are not computing average here, just the sum
    return (y_hat - tf.reshape(y, y_hat.shape))**2 / 2 # reshape just in case, sometimes y y_hat could be either (1,n) or (n,1)

def sgd(params, grads, lr, batch_size):
    """
    mini-batch SGD
    """
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / batch_size) # since we are not computing average in the last step so we devide by batch size here

def data_iter(batch_size, features, labels):
    """
    generate a data iter which takes the input (batch, features, labels)
    output: mini batch of (X,y)
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i:min(i + batch_size, num_examples)]) # handle the last batch whose size may be smaller than batch_size
        yield tf.gather(features, j), tf.gather(labels, j)


if __name__ == '__main__':

    # generate synthetic data
    true_w = tf.constant([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # init
    w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01), trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # start to train the linear regression model
    lr = 0.03 # learning rate
    num_epochs = 3
    batch_size = 10


    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            with tf.GradientTape() as g:
                l = squared_loss(linreg(X, w, b), y)  # mini-batch loss of `X`和`y`
            # compute the gradients of l w.r.t [`w`, `b`]
            dw, db = g.gradient(l, [w, b])
            # update parameters
            sgd([w, b], [dw, db], lr, batch_size) # not entirely correct cuz the last batch has size <= batch_size, but ignore the nuance for now

        train_l = squared_loss(linreg(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')

    # print out the distance between the estimated parameters and the true parameters
    print(f'diff in w: {true_w - tf.reshape(w, true_w.shape)}')
    print(f'diff in b: {true_b - b}')
