import numpy as np
import random
import tensorflow as tf


def get_negative_mask(b_size):
    # remove similarity score of similar cascades
    # codes from https://github.com/sthalles/SimCLR-tensorflow/blob/master/utils/helpers.py
    negative_mask = np.ones((b_size, b_size * 2), dtype=bool)
    for i in range(b_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + b_size] = 0
    return tf.constant(negative_mask)


def dot_sim_1(x, y):
    # codes from https://github.com/sthalles/SimCLR-tensorflow/blob/master/utils/losses.py
    return tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))


def dot_sim_2(x, y):
    # codes from https://github.com/sthalles/SimCLR-tensorflow/blob/master/utils/losses.py
    return tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)


def shuffle_two(x, y):
    couple = list(zip(x, y))
    random.shuffle(couple)
    return zip(*couple)


def divide_dataset(x, label_fractions=100):
    # only for 1%, 10%, and 100% label fractions
    return x[::100//label_fractions]
