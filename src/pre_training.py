import copy
import pickle
import time

import numpy as np
import tensorflow as tf
from absl import app, flags
from utils.tools import dot_sim_1, dot_sim_2  # similarity functions
from utils.tools import get_negative_mask, shuffle_two

# let gpu do not take up all of the gpu memory
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

print('TF Version:', tf.__version__)

# flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, 'Mini-batch size.')
flags.DEFINE_integer('emb_dim', 64, 'Embedding dimension.')
flags.DEFINE_integer('max_seq', 100, 'Max length of cascade sequence.')
flags.DEFINE_float(  'l2', 5e-4, 'L2 coefficient.')
flags.DEFINE_float(  'lr', 5e-4, 'Learning rate.')
flags.DEFINE_integer('model_size', 4, 'Model size.')
flags.DEFINE_integer('model_size_scale', 32, 'Model size base scale.')
flags.DEFINE_string( 'name', 'xovee', 'Name of this experiment.')
flags.DEFINE_integer('pre_training_epochs', 30, 'Pre-training epochs.')
flags.DEFINE_string( 'projection_head', '2-0', 'MLP-based projection head.')
flags.DEFINE_float(  'temperature', .1, 'Hyper-parameter temperature for contrastive loss.')
flags.DEFINE_boolean('use_unlabel', True, 'Pre-training with unlabeled data.')

# paths
flags.DEFINE_string('input', './datasets/weibo/', 'Pre-training data path.')
flags.DEFINE_string('weight_path', './results/pre_training_weight/', 'Path of saved encoder weights.')


def main(argv):
    start_time = time.time()

    # hyper-params
    batch_size = FLAGS.batch_size
    emb_dim = FLAGS.emb_dim
    gru_units = FLAGS.model_size * FLAGS.model_size_scale
    epochs = FLAGS.pre_training_epochs
    l2 = FLAGS.l2
    lr = FLAGS.lr
    max_seq = FLAGS.max_seq
    projection_head = FLAGS.projection_head
    temperature = FLAGS.temperature
    # hyper-params

    # build model
    inputs = tf.keras.layers.Input(shape=(max_seq, emb_dim))

    gru_1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(gru_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2))
    )(inputs)
    gru_2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(gru_units, kernel_regularizer=tf.keras.regularizers.l2(l2))
    )(gru_1)

    # MLP-based projection head
    if projection_head[0] == '0':
        encoder = tf.keras.models.Model(inputs, gru_2, name='encoder')
        encoder_projection = tf.keras.models.Model(inputs, gru_2, name='encoder-projection')
    elif projection_head[0] == '1':
        mlp = tf.keras.layers.Dense(gru_units*2, activation='linear')(gru_2)
        if projection_head[2] == '0':
            encoder = tf.keras.models.Model(inputs, gru_2, name='encoder')
            encoder_projection = tf.keras.models.Model(inputs, mlp, name='encoder-projection')
        elif projection_head[2] == '1':
            encoder = tf.keras.models.Model(inputs, mlp, name='encoder')
            encoder_projection = tf.keras.models.Model(inputs, mlp, name='encoder-projection')
        else:
            print('Wrong projection head argument, should be [0-4]-[0-4].')
    elif projection_head[0] == '2':
        mlp_1 = tf.keras.layers.Dense(gru_units * 2, activation='relu')(gru_2)
        mlp_2 = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp_1)
        if projection_head[2] == '0':
            encoder = tf.keras.models.Model(inputs, gru_2, name='encoder')
            encoder_projection = tf.keras.models.Model(inputs, mlp_2, name='encoder-projection')
        elif projection_head[2] == '1':
            encoder = tf.keras.models.Model(inputs, mlp_1, name='encoder')
            encoder_projection = tf.keras.models.Model(inputs, mlp_2, name='encoder-projection')
        elif projection_head[2] == '2':
            encoder = tf.keras.models.Model(inputs, mlp_2, name='encoder')
            encoder_projection = tf.keras.models.Model(inputs, mlp_2, name='encoder-projection')
        else:
            print('Wrong projection head argument, should be [0-4]-[0-4].')
    elif projection_head[0] == '3':
        mlp_1 = tf.keras.layers.Dense(gru_units * 2, activation='relu')(gru_2)
        mlp_2 = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp_1)
        mlp_3 = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp_2)
        if projection_head[2] == '0':
            encoder = tf.keras.models.Model(inputs, gru_2, name='encoder')
            encoder_projection = tf.keras.models.Model(inputs, mlp_3, name='encoder-projection')
        elif projection_head[2] == '1':
            encoder = tf.keras.models.Model(inputs, mlp_1, name='encoder')
            encoder_projection = tf.keras.models.Model(inputs, mlp_3, name='encoder-projection')
        elif projection_head[2] == '2':
            encoder = tf.keras.models.Model(inputs, mlp_2, name='encoder')
            encoder_projection = tf.keras.models.Model(inputs, mlp_3, name='encoder-projection')
        elif projection_head[2] == '3':
            encoder = tf.keras.models.Model(inputs, mlp_3, name='encoder')
            encoder_projection = tf.keras.models.Model(inputs, mlp_3, name='encoder-projection')
        else:
            print('Wrong projection head argument, should be [0-4]-[0-4].')
    elif projection_head[0] == '4':
        mlp_1 = tf.keras.layers.Dense(gru_units * 2, activation='relu')(gru_2)
        mlp_2 = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp_1)
        mlp_3 = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp_2)
        mlp_4 = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp_3)
        if projection_head[2] == '0':
            encoder = tf.keras.models.Model(inputs, gru_2, name='encoder')
            encoder_projection = tf.keras.models.Model(inputs, mlp_4, name='encoder-projection')
        elif projection_head[2] == '1':
            encoder = tf.keras.models.Model(inputs, mlp_1, name='encoder')
            encoder_projection = tf.keras.models.Model(inputs, mlp_4, name='encoder-projection')
        elif projection_head[2] == '2':
            encoder = tf.keras.models.Model(inputs, mlp_2, name='encoder')
            encoder_projection = tf.keras.models.Model(inputs, mlp_4, name='encoder-projection')
        elif projection_head[2] == '3':
            encoder = tf.keras.models.Model(inputs, mlp_3, name='encoder')
            encoder_projection = tf.keras.models.Model(inputs, mlp_4, name='encoder-projection')
        elif projection_head[2] == '4':
            encoder = tf.keras.models.Model(inputs, mlp_4, name='encoder')
            encoder_projection = tf.keras.models.Model(inputs, mlp_4, name='encoder-projection')
        else:
            print('Wrong projection head argument, should be [0-4]-[0-4].')
    else:
        print("Wrong projection head argument, should be [0-4]-[0-4].")

    encoder_projection.summary()

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # negative mask
    negative_mask = get_negative_mask(batch_size)

    # criterion
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

    # loss
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(data_1, data_2):
        with tf.GradientTape() as tape:
            z_1 = encoder_projection(data_1)
            z_2 = encoder_projection(data_2)

            z_1 = tf.math.l2_normalize(z_1, axis=1)
            z_2 = tf.math.l2_normalize(z_2, axis=1)

            l_pos = tf.reshape(dot_sim_1(z_1, z_2), (batch_size, 1)) / temperature

            negatives = tf.concat([z_1, z_2], axis=0)

            loss = 0

            for positives in [z_1, z_2]:
                l_neg = dot_sim_2(positives, negatives)

                labels = tf.zeros(batch_size, dtype=tf.int32)

                l_neg = tf.reshape(tf.boolean_mask(l_neg, negative_mask), (batch_size, -1)) / temperature

                logits = tf.concat([l_pos, l_neg], axis=1)
                loss += criterion(y_pred=logits, y_true=labels)

            loss = loss / (2 * batch_size)

        gradients = tape.gradient(loss, encoder_projection.trainable_variables)
        optimizer.apply_gradients(zip(gradients, encoder_projection.trainable_variables))

        train_loss(loss)

    # load data
    with open(FLAGS.input + 'data_aug_1.pkl', 'rb') as f:
        aug_1 = pickle.load(f)
    with open(FLAGS.input + 'data_aug_2.pkl', 'rb') as f:
        aug_2 = pickle.load(f)
    if FLAGS.use_unlabel:
        with open(FLAGS.input + 'data_unlabel_aug_1.pkl', 'rb') as f:
            unlabeled_aug_1 = pickle.load(f)
            aug_1.extend(unlabeled_aug_1)
        with open(FLAGS.input + 'data_unlabel_aug_1.pkl', 'rb') as f:
            unlabeled_aug_2 = pickle.load(f)
            aug_2.extend(unlabeled_aug_2)

    # print data information
    dataset_info = '# pre-training samples {}'
    print(dataset_info.format(len(aug_1)))

    # pre-training
    best_train_loss = 1000
    for epoch in range(epochs):
        time_start = time.time()
        train_loss.reset_states()
        aug_1, aug_2 = shuffle_two(aug_1, aug_2)

        for i in range(len(aug_1)//batch_size):
            batch_aug_1 = copy.deepcopy(aug_1[batch_size * i:batch_size * i + batch_size])
            batch_aug_2 = copy.deepcopy(aug_2[batch_size * i:batch_size * i + batch_size])
            for batch_cascade_1 in batch_aug_1:
                while len(batch_cascade_1) < max_seq:
                    batch_cascade_1.append(np.zeros(emb_dim))
            for batch_cascade_2 in batch_aug_2:
                while len(batch_cascade_2) < max_seq:
                    batch_cascade_2.append(np.zeros(emb_dim))
            train_step(np.array(batch_aug_1), np.array(batch_aug_2))

        time_now = time.strftime("%Y-%m-%d, %H:%M", time.localtime())
        template = '{}: Pre-training Epoch {:3}, Time: {:.3f}s, {}, Train Loss: {:.4f}'
        print(template.format(FLAGS.name, epoch + 1, time.time() - time_start, time_now, train_loss.result()))

        if train_loss.result() < best_train_loss:
            best_train_loss = train_loss.result()

            encoder.save_weights(FLAGS.weight_path + FLAGS.name + '.h5', save_format='h5')
            print('Model saved!')

    print('Finished! Time used: {:.3f}min'.format((time.time()-start_time)/60))


if __name__ == '__main__':
    app.run(main)
