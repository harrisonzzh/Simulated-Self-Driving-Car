import tensorflow as tf
import numpy as np

def weight_variable(shape):
    # shape [window dim1, window dim2, depth1, depth2]
    weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return weight


def bias_variable(depth):
    return tf.Variable(tf.zeros([depth]))


def conv2d(data, weights, stride):
    # data: training data
    # stride: window moving step. (datatype: int)
    conv_layer = tf.nn.conv2d(data, weights, strides=[1, stride, stride, 1], padding="VALID")
    return conv_layer


def img_norm(img):
    # Image normalization to avoid saturation and make gradients work better.
    return img / 127.5 - 1.0


def accuracy(predictions, labels):
    return (np.sum((predictions - labels) ** 2) ** 0.5) / len(labels)


# input data
X = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
y = tf.placeholder(tf.float32, shape=[None])

# Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
W_conv1 = weight_variable([5, 5, 3, 24])
b_conv1 = bias_variable(24)

# Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
W_conv2 = weight_variable([5, 5, 24, 36])
b_conv2 = bias_variable(36)

# Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
W_conv3 = weight_variable([5, 5, 36, 48])
b_conv3 = bias_variable(48)

# Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
W_conv4 = weight_variable([3, 3, 48, 64])
b_conv4 = bias_variable(64)

# Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
W_conv5 = weight_variable([3, 3, 64, 64])
b_conv5 = bias_variable(64)

# dropout probability
keep_prob = tf.placeholder(tf.float32)

# Fully connected: neurons: 1164, activation: ELU
W_fc1 = weight_variable([1152, 1164])
b_fc1 = bias_variable(1164)

# Fully connected: neurons: 100, activation: ELU
W_fc2 = weight_variable([1164, 100])
b_fc2 = bias_variable(100)

# Fully connected: neurons: 50, activation: ELU
W_fc3 = weight_variable([100, 50])
b_fc3 = bias_variable(50)

# Fully connected: neurons: 10, activation: ELU
W_fc4 = weight_variable([50, 10])
b_fc4 = bias_variable(10)

# Fully connected: neurons: 1 (output)
W_fc5 = weight_variable([10, 1])
b_fc5 = bias_variable(1)

# Placeholder for validation
valid_status = tf.placeholder(tf.int32)


def model(data):
    conv1 = tf.nn.relu(conv2d((data), W_conv1, 2) + b_conv1)

    conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 2) + b_conv2)

    conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 2) + b_conv3)

    conv4 = tf.nn.relu(conv2d(conv3, W_conv4, 1) + b_conv4)

    conv5 = tf.nn.relu(conv2d(conv4, W_conv5, 1) + b_conv5)
    conv5_flat = tf.reshape(conv5, [-1, 1152])

    fc1 = tf.nn.relu(tf.matmul(conv5_flat, W_fc1) + b_fc1)
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    fc2 = tf.nn.relu(tf.matmul(fc1_drop, W_fc2) + b_fc2)
    fc2_drop = tf.nn.dropout(fc2, keep_prob)

    fc3 = tf.nn.relu(tf.matmul(fc2_drop, W_fc3) + b_fc3)
    fc3_drop = tf.nn.dropout(fc3, keep_prob)

    fc4 = tf.nn.relu(tf.matmul(fc3_drop, W_fc4) + b_fc4)
    fc4_drop = tf.nn.dropout(fc4, keep_prob)

    y = tf.multiply(tf.atan(tf.matmul(fc4_drop, W_fc5) + b_fc5), 2) #scale the atan output
    return y


# Training computation
y_ = model(X)