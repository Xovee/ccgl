import copy
import pickle
import time

import numpy as np
import tensorflow as tf
from absl import app, flags
from utils.tools import divide_dataset, shuffle_two

# let gpu do not take up all of the gpu memory
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

print('TF Version:', tf.__version__)

# flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, 'Mini-batch size.')
flags.DEFINE_integer('emb_dim', 64, 'Embedding dimension.')
flags.DEFINE_integer('epochs', 1000, 'Pre-training epochs.')
flags.DEFINE_boolean('freeze', False, 'Linear evaluation on frozen features.')
flags.DEFINE_integer('max_seq', 100, 'Max length of cascade sequence.')
flags.DEFINE_float(  'l2', 5e-4, 'L2 coefficient.')
flags.DEFINE_integer('label_fraction', 100, 'Label fraction, only for 1%, 10%, and 100%.')
flags.DEFINE_float(  'lr', 5e-4, 'Learning rate.')
flags.DEFINE_integer('model_size', 4, 'Model size.')
flags.DEFINE_integer('model_size_scale', 32, 'Model size base scale.')
flags.DEFINE_string( 'name', 'n999', 'Name of this experiment.')
flags.DEFINE_string( 'num', '0', 'A number in saved path of teacher weights.')
flags.DEFINE_integer('patience', 20, 'Patience for early stopping.')
flags.DEFINE_string( 'projection_head', '2-0', 'MLP-based projection head.')

# paths
flags.DEFINE_string('input', './dataset/weibo/', 'Pre-training data path.')
flags.DEFINE_string('result_path', './results/prediction/', 'Path of model predictions.')
flags.DEFINE_string('weight_path', './results/pre_training_weight/', 'Path of saved encoder weights.')
flags.DEFINE_string('teacher_path', './results/fine_tuning_weight/', 'Path of teacher network weights.')


def main(argv):
    start_time = time.time()

    # hyper-params
    batch_size = FLAGS.batch_size
    emb_dim = FLAGS.emb_dim
    gru_units = FLAGS.model_size * FLAGS.model_size_scale
    epochs = FLAGS.epochs
    freeze = FLAGS.freeze
    l2 = FLAGS.l2
    label_fraction = FLAGS.label_fraction
    lr = FLAGS.lr
    max_seq = FLAGS.max_seq
    patience = FLAGS.patience
    projection_head = FLAGS.projection_head
    # hyper-params

    # build model
    inputs = tf.keras.layers.Input(shape=(max_seq, emb_dim))

    gru_1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(gru_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2))
    )(inputs)
    gru_2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(gru_units, kernel_regularizer=tf.keras.regularizers.l2(l2))
    )(gru_1)

    # build projection head
    if projection_head[2] == '0':
        encoder = tf.keras.models.Model(inputs, gru_2, name='encoder')
        encoder.load_weights(FLAGS.weight_path + FLAGS.name + '.h5')
        mlp_1 = tf.keras.layers.Dense(gru_units, activation='relu')(gru_2)
        mlp_2 = tf.keras.layers.Dense(gru_units // 2, activation='relu')(mlp_1)
        prediction = tf.keras.layers.Dense(1)(mlp_2)
    elif projection_head[2] == '1':
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='linear')(gru_2)
        encoder = tf.keras.models.Model(inputs, mlp, name='encoder')
        encoder.load_weights(FLAGS.weight_path + FLAGS.name + '.h5')
        mlp_1 = tf.keras.layers.Dense(gru_units, activation='relu')(mlp)
        mlp_2 = tf.keras.layers.Dense(gru_units // 2, activation='relu')(mlp_1)
        prediction = tf.keras.layers.Dense(1)(mlp_2)
    elif projection_head[2] == '2':
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(gru_2)
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp)
        encoder = tf.keras.models.Model(inputs, mlp, name='encoder')
        encoder.load_weights(FLAGS.weight_path + FLAGS.name + '.h5')
        mlp_1 = tf.keras.layers.Dense(gru_units, activation='relu')(mlp)
        mlp_2 = tf.keras.layers.Dense(gru_units // 2, activation='relu')(mlp_1)
        prediction = tf.keras.layers.Dense(1)(mlp_2)
    elif projection_head[2] == '3':
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(gru_2)
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp)
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp)
        encoder = tf.keras.models.Model(inputs, mlp, name='encoder')
        encoder.load_weights(FLAGS.weight_path + FLAGS.name + '.h5')
        mlp_1 = tf.keras.layers.Dense(gru_units, activation='relu')(mlp)
        mlp_2 = tf.keras.layers.Dense(gru_units // 2, activation='relu')(mlp_1)
        prediction = tf.keras.layers.Dense(1)(mlp_2)
    elif projection_head[2] == '4':
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(gru_2)
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp)
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp)
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp)
        encoder = tf.keras.models.Model(inputs, mlp, name='encoder')
        encoder.load_weights(FLAGS.weight_path + FLAGS.name + '.h5')
        mlp_1 = tf.keras.layers.Dense(gru_units, activation='relu')(mlp)
        mlp_2 = tf.keras.layers.Dense(gru_units // 2, activation='relu')(mlp_1)
        prediction = tf.keras.layers.Dense(1)(mlp_2)
    else:
        print('Wrong projection head argument, should be [0-4]-[0-4].')

    # freeze the encoder or not
    if FLAGS.freeze:
        encoder.trainable = False

    fine_tuning_model = tf.keras.models.Model(inputs, prediction, name='fine_tuning_model')
    fine_tuning_model.summary()

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # loss
    loss_object = tf.keras.losses.MeanSquaredLogarithmicError()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    @tf.function
    def train_step(data, labels):
        with tf.GradientTape() as tape:
            predictions = fine_tuning_model(data, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, fine_tuning_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, fine_tuning_model.trainable_variables))

        train_loss(loss)

    @tf.function
    def val_step(data, labels):
        predictions = fine_tuning_model(data, training=False)
        v_loss = loss_object(labels, predictions)

        val_loss(v_loss)

    @tf.function
    def test_step(data, labels):
        predictions = fine_tuning_model(data, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)

        return predictions

    # load data
    with open(FLAGS.input + 'train.pkl', 'rb') as f:
        train, train_y = pickle.load(f)
    with open(FLAGS.input + 'val.pkl', 'rb') as f:
        val, val_y = pickle.load(f)
    with open(FLAGS.input + 'test.pkl', 'rb') as f:
        test, test_y = pickle.load(f)

    # divide dataset
    train, train_y = divide_dataset(train, label_fraction), divide_dataset(train_y, label_fraction)

    # print data information
    dataset_info = '# fine-tuning samples {}\n' + \
                   '#  validation samples {}\n' + \
                   '#        test samples {}'
    print(dataset_info.format(len(train), len(val), len(test)))

    # linear evaluation or fine-tuning
    best_val_loss = 1000
    save_predictions = list()
    for epoch in range(epochs):
        time_start = time.time()
        train_loss.reset_states()
        val_loss.reset_states()
        test_loss.reset_states()
        train, train_y = shuffle_two(train, train_y)

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

        template = '{}: Fine-tuning Epoch {:3}, Time: {:.3f}s, Train Loss: {:.3f}, Val Loss: {:.3f}, ' \
                   'Test Loss: {:.3f}, LOG2 MSLE: {:.3f}'
        print(template.format(FLAGS.name, epoch + 1, time.time() - time_start,
                              train_loss.result(), val_loss.result(), test_loss.result(), report_loss))

        if val_loss.result() < best_val_loss:
            best_val_loss = val_loss.result()
            save_predictions = pred
            patience = FLAGS.patience

            # save model
            fine_tuning_model.save_weights(FLAGS.teacher_path + FLAGS.name + '-' + FLAGS.num + '.h5')
            print('Model saved!')

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
