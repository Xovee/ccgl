import copy
import pickle
import time

import numpy as np
import tensorflow as tf
from absl import app, flags
from utils.tools import divide_dataset, shuffle_two

# flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, 'Mini-batch size.')
flags.DEFINE_integer('emb_dim', 64, 'Embedding dimension.')
flags.DEFINE_integer('epochs', 1000, 'Training epochs.')
flags.DEFINE_float(  'l2', 5e-4, 'L2 coefficient.')
flags.DEFINE_integer('label_fraction', 100, 'Label fraction, only for 1%, 10%, and 100%.')
flags.DEFINE_float(  'lr', 5e-4, 'Learning rate.')
flags.DEFINE_integer('max_seq', 100, 'Max length of cascade sequence.')
flags.DEFINE_integer('model_size', 4, 'Model size.')
flags.DEFINE_integer('model_size_scale', 32, 'Model size base scale.')
flags.DEFINE_string( 'name', 'xovee', 'Name of this run.')
flags.DEFINE_integer('patience', 20, 'Patience for early stopping.')

# paths
flags.DEFINE_string( 'input', './datasets/weibo/', 'Pre-training data path.')

# let gpu do not take up all of the gpu memory
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)


start_time = time.time()
print('TF Version:', tf.__version__)


def main(argv):
    # hyper-params
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    gru_units = FLAGS.model_size * FLAGS.model_size_scale
    emb_dim = FLAGS.emb_dim
    max_seq = FLAGS.max_seq
    patience = FLAGS.patience
    l2 = FLAGS.l2
    lr = FLAGS.lr
    # hyper-params

    # load data
    with open(FLAGS.input + 'train.pkl', 'rb') as f:
        train, train_y = pickle.load(f)
    with open(FLAGS.input + 'val.pkl', 'rb') as f:
        val, val_y = pickle.load(f)
    with open(FLAGS.input + 'test.pkl', 'rb') as f:
        test, test_y = pickle.load(f)

    train = divide_dataset(train, label_fractions=FLAGS.label_fraction)
    train_y = divide_dataset(train_y, label_fractions=FLAGS.label_fraction)

    # print data information
    dataset_info = '#   training samples {}\n' + \
                   '# validation samples {}\n' + \
                   '#       test samples {}'
    print(dataset_info.format(len(train), len(val), len(test)))

    # build model
    inputs = tf.keras.layers.Input(shape=(max_seq, emb_dim))

    gru_1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(gru_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2))
    )(inputs)
    gru_2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(gru_units, kernel_regularizer=tf.keras.regularizers.l2(l2))
    )(gru_1)

    mlp_1 = tf.keras.layers.Dense(gru_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2))(gru_2)
    mlp_2 = tf.keras.layers.Dense(gru_units//2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2))(mlp_1)

    outputs = tf.keras.layers.Dense(1)(mlp_2)

    base_model = tf.keras.models.Model(inputs, outputs)
    base_model.summary()

    # optimizer
    optimizer = tf.keras.optimizers.Adam(lr)

    # loss
    loss_object = tf.keras.losses.MeanSquaredLogarithmicError()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_msle = tf.keras.metrics.MeanSquaredLogarithmicError(name='train_msle')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_msle = tf.keras.metrics.MeanSquaredLogarithmicError(name='train_msle')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_msle = tf.keras.metrics.MeanSquaredLogarithmicError(name='test_msle')

    @tf.function
    def train_step(data, labels):
        with tf.GradientTape() as tape:
            predictions = base_model(data, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, base_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, base_model.trainable_variables))

        train_loss(loss)
        train_msle(labels, predictions)

    @tf.function
    def val_step(data, labels):
        predictions = base_model(data, training=False)
        v_loss = loss_object(labels, predictions)

        val_loss(v_loss)
        val_msle(labels, predictions)

    @tf.function
    def test_step(data, labels):
        predictions = base_model(data, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_msle(labels, predictions)

        return predictions

    # training
    best_val_msle = 1000
    save_predictions = list()
    for epoch in range(epochs):
        train, train_y = shuffle_two(train, train_y)
        time_start = time.time()
        train_loss.reset_states()
        train_msle.reset_states()
        val_loss.reset_states()
        val_msle.reset_states()
        test_loss.reset_states()
        test_msle.reset_states()

        for i in range(len(train)//batch_size+1):
            batch_train = copy.deepcopy(train[batch_size * i:batch_size * i + batch_size])
            batch_train_labels = train_y[batch_size * i:batch_size * i + batch_size]
            for batch_cascade in batch_train:
                while len(batch_cascade) < max_seq:
                    batch_cascade.append(np.zeros(emb_dim))
            train_step(np.array(batch_train), np.array(batch_train_labels))

        for i in range(len(val)//batch_size+1):
            batch_val = copy.deepcopy(val[batch_size * i:batch_size * i + batch_size])
            batch_val_labels = val_y[batch_size * i:batch_size * i + batch_size]
            for batch_cascade in batch_val:
                while len(batch_cascade) < max_seq:
                    batch_cascade.append(np.zeros(emb_dim))
            val_step(np.array(batch_val), np.array(batch_val_labels))

        pred = list()
        for i in range(len(test)//batch_size+1):
            batch_test = copy.deepcopy(test[batch_size * i:batch_size * i + batch_size])
            batch_test_labels = test_y[batch_size * i:batch_size * i + batch_size]

            for batch_cascade in batch_test:
                while len(batch_cascade) < max_seq:
                    batch_cascade.append(np.zeros(emb_dim))
            batch_predictions = test_step(np.array(batch_test), np.array(batch_test_labels))
            pred.extend(batch_predictions)

        pred = [float(pre) for pre in pred]
        report_loss = np.mean(np.square(np.log2(np.array([pre if pre >= 1 else 1 for pre in pred])) -
                              np.log2(np.array([tru if tru >= 1 else 1 for tru in list(test_y)]))))

        if val_msle.result() < best_val_msle:
            best_val_msle = val_msle.result()
            save_predictions = pred
            patience = FLAGS.patience
            template = 'Epoch {:2}, Time: {:.3f}s, ' \
                       'Train Loss: {:.3f}, Train MSLE: {:.3f}, ' \
                       'Val Loss: {:.3f}, Val MSLE: {:.3f}, ' \
                       'Test Loss: {:.3f}, Test MSLE: {:.3f}, LOG2 MSLE: {:.3f}'
            print(template.format(epoch + 1, time.time()-time_start,
                                  train_loss.result(),
                                  train_msle.result(),
                                  val_loss.result(),
                                  val_msle.result(),
                                  test_loss.result(),
                                  test_msle.result(),
                                  report_loss))

        if patience == 0:
            report_loss = np.mean(np.square(np.log2(np.array([pre if pre >= 1 else 1 for pre in save_predictions])) -
                                            np.log2(np.array([tru if tru >= 1 else 1 for tru in list(test_y)]))))

            print('Predictions saved! Best Test MSLE: {}'.format(report_loss))
            break
        else:
            patience -= 1

    print('Finished! Time used: {:.3f}min'.format((time.time()-start_time)/60))


if __name__ == '__main__':
    app.run(main)
