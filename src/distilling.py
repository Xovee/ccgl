import copy
import pickle
import random
import time

import numpy as np
import tensorflow as tf
from absl import app, flags

start_time = time.time()
print('TF Version:', tf.__version__)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

# flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, 'Mini-batch size.')
flags.DEFINE_boolean('distill_with_unlabel', True, 'Distill with label.')
flags.DEFINE_integer('emb_dim', 64, 'Embedding dimension.')
flags.DEFINE_integer('epochs', 1000, 'Distillation epochs.')
flags.DEFINE_float(  'l2', 5e-4, 'L2 coefficient.')
flags.DEFINE_float(  'lr', 5e-4, 'Learning rate.')
flags.DEFINE_integer('max_seq', 100, 'Max length of cascade sequence.')
flags.DEFINE_integer('model_size', 4, 'Model size.')
flags.DEFINE_integer('model_size_scale', 32, 'Model size base scale.')
flags.DEFINE_string( 'name', 'xovee', 'Name of this experiment.')
flags.DEFINE_string( 'num', '0', 'A number in path to load teacher weight and save student weight.')
flags.DEFINE_integer('patience', 20, 'Patience for early stopping.')
flags.DEFINE_string( 'projection_head', '2-0', 'MLP-based projection head.')
flags.DEFINE_boolean('self_distill', True, 'Self distillation.')

# paths
flags.DEFINE_string('input', './datasets/weibo/', 'Distillation data path.')
flags.DEFINE_string('result_path', './results/prediction/', 'Path of model predictions.')
flags.DEFINE_string('teacher_path', './results/fine_tuning_weight/', 'Path of saved teacher weigths.')
flags.DEFINE_string('student_path', './results/student_weight/', 'Path of saved student weigths.')


def main(argv):
    # hyper-params
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    gru_units = FLAGS.model_size * FLAGS.model_size_scale
    max_seq = FLAGS.max_seq
    emb_dim = FLAGS.emb_dim
    patience = FLAGS.patience
    l2 = FLAGS.l2
    lr = FLAGS.lr
    projection_head = FLAGS.projection_head
    # hyper-params

    # load data
    with open(FLAGS.input + 'train.pkl', 'rb') as f:
        train, _ = pickle.load(f)
    with open(FLAGS.input + 'val.pkl', 'rb') as f:
        val, val_y = pickle.load(f)
    with open(FLAGS.input + 'test.pkl', 'rb') as f:
        test, test_y = pickle.load(f)

    if FLAGS.distill_with_unlabel:
        with open(FLAGS.input + 'data_unlabel_aug_1.pkl', 'rb') as f:
            unlabel_aug_1 = pickle.load(f)
        with open(FLAGS.input + 'data_unlabel_aug_2.pkl', 'rb') as f:
            unlabel_aug_2 = pickle.load(f)
        unlabel_aug_1.extend(unlabel_aug_2)
        train.extend(unlabel_aug_1)

    # print data information
    dataset_info = '#   unlabeled samples {}\n' + \
                   '#  validation samples {}\n' + \
                   '#        test samples {}'
    print(dataset_info.format(len(train), len(val), len(test)))

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
        mlp_1 = tf.keras.layers.Dense(gru_units, activation='relu')(gru_2)
        mlp_2 = tf.keras.layers.Dense(gru_units // 2, activation='relu')(mlp_1)
        prediction = tf.keras.layers.Dense(1)(mlp_2)
    elif projection_head[2] == '1':
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='linear')(gru_2)
        mlp_1 = tf.keras.layers.Dense(gru_units, activation='relu')(mlp)
        mlp_2 = tf.keras.layers.Dense(gru_units // 2, activation='relu')(mlp_1)
        prediction = tf.keras.layers.Dense(1)(mlp_2)
    elif projection_head[2] == '2':
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(gru_2)
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp)
        mlp_1 = tf.keras.layers.Dense(gru_units, activation='relu')(mlp)
        mlp_2 = tf.keras.layers.Dense(gru_units // 2, activation='relu')(mlp_1)
        prediction = tf.keras.layers.Dense(1)(mlp_2)
    elif projection_head[2] == '3':
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(gru_2)
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp)
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp)
        mlp_1 = tf.keras.layers.Dense(gru_units, activation='relu')(mlp)
        mlp_2 = tf.keras.layers.Dense(gru_units // 2, activation='relu')(mlp_1)
        prediction = tf.keras.layers.Dense(1)(mlp_2)
    elif projection_head[2] == '4':
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(gru_2)
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp)
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp)
        mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(mlp)
        mlp_1 = tf.keras.layers.Dense(gru_units, activation='relu')(mlp)
        mlp_2 = tf.keras.layers.Dense(gru_units // 2, activation='relu')(mlp_1)
        prediction = tf.keras.layers.Dense(1)(mlp_2)
    else:
        print('Wrong projection head argument, should be [0-4]-[0-4].')

    teacher = tf.keras.models.Model(inputs, prediction, name='teacher_model')
    teacher.load_weights(FLAGS.teacher_path + FLAGS.name + '.h5')
    teacher.trainable = False

    student_inputs = tf.keras.layers.Input(shape=(max_seq, emb_dim))

    # build a self-distilled student or a dense student
    if FLAGS.self_distill:
        student_gru_1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(gru_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2))
        )(student_inputs)
        student_gru_2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(gru_units, kernel_regularizer=tf.keras.regularizers.l2(l2))
        )(student_gru_1)

        if projection_head[2] == '0':
            student_mlp_1 = tf.keras.layers.Dense(gru_units, activation='relu')(student_gru_2)
            student_mlp_2 = tf.keras.layers.Dense(gru_units // 2, activation='relu')(student_mlp_1)
            student_prediction = tf.keras.layers.Dense(1)(student_mlp_2)
        elif projection_head[2] == '1':
            student_mlp = tf.keras.layers.Dense(gru_units * 2, activation='linear')(student_gru_2)
            student_mlp_1 = tf.keras.layers.Dense(gru_units, activation='relu')(student_mlp)
            student_mlp_2 = tf.keras.layers.Dense(gru_units // 2, activation='relu')(student_mlp_1)
            student_prediction = tf.keras.layers.Dense(1)(student_mlp_2)
        elif projection_head[2] == '2':
            student_mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(student_gru_2)
            student_mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(student_mlp)
            student_mlp_1 = tf.keras.layers.Dense(gru_units, activation='relu')(student_mlp)
            student_mlp_2 = tf.keras.layers.Dense(gru_units // 2, activation='relu')(student_mlp_1)
            student_prediction = tf.keras.layers.Dense(1)(student_mlp_2)
        elif projection_head[2] == '3':
            student_mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(student_gru_2)
            student_mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(student_mlp)
            student_mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(student_mlp)
            student_mlp_1 = tf.keras.layers.Dense(gru_units, activation='relu')(student_mlp)
            student_mlp_2 = tf.keras.layers.Dense(gru_units // 2, activation='relu')(student_mlp_1)
            student_prediction = tf.keras.layers.Dense(1)(student_mlp_2)
        elif projection_head[2] == '4':
            student_mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(student_gru_2)
            student_mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(student_mlp)
            student_mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(student_mlp)
            student_mlp = tf.keras.layers.Dense(gru_units * 2, activation='relu')(student_mlp)
            student_mlp_1 = tf.keras.layers.Dense(gru_units, activation='relu')(student_mlp)
            student_mlp_2 = tf.keras.layers.Dense(gru_units // 2, activation='relu')(student_mlp_1)
            student_prediction = tf.keras.layers.Dense(1)(student_mlp_2)
        else:
            print('Wrong projection head argument, should be [0-4]-[0-4].')

        student = tf.keras.models.Model(student_inputs, student_prediction, name='student')
        student.trainable = True
    else:
        student_gru = tf.keras.layers.GRU(gru_units, kernel_regularizer=tf.keras.regularizers.l2(l2))(student_inputs)
        student_mlp = tf.keras.layers.Dense(gru_units // 2, activation='relu')(student_gru)
        student_prediction = tf.keras.layers.Dense(1)(student_mlp)
        student = tf.keras.models.Model(student_inputs, student_prediction, name='student')
        student.trainable = True

    # summary
    teacher.summary()
    student.summary()

    # optimizer
    optimizer = tf.keras.optimizers.Adam(lr)

    # loss
    loss_object = tf.keras.losses.MeanSquaredLogarithmicError()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    @tf.function
    def train_step(data):
        with tf.GradientTape() as tape:
            predictions = teacher(data, training=False)
            student_predictions = student(data, training=True)
            loss = loss_object(predictions, student_predictions)
        gradients = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(gradients, student.trainable_variables))

        train_loss(loss)

    @tf.function
    def val_step(data, labels):
        predictions = student(data, training=False)
        v_loss = loss_object(labels, predictions)

        val_loss(v_loss)

    @tf.function
    def test_step(data, labels):
        predictions = student(data, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)

        return predictions

    # distilling
    best_val_loss = 1000
    save_predictions = list()
    for epoch in range(epochs):
        time_start = time.time()
        train_loss.reset_states()
        val_loss.reset_states()
        test_loss.reset_states()
        random.shuffle(train)

        for i in range(len(train) // batch_size + 1):
            batch_train = copy.deepcopy(train[batch_size * i:batch_size * i + batch_size])
            for batch_cascade in batch_train:
                while len(batch_cascade) < max_seq:
                    batch_cascade.append(np.zeros(emb_dim))
            train_step(np.array(batch_train))

        for i in range(len(val) // batch_size + 1):
            batch_val = copy.deepcopy(val[batch_size * i:batch_size * i + batch_size])
            batch_val_labels = val_y[batch_size * i:batch_size * i + batch_size]
            for batch_cascade in batch_val:
                while len(batch_cascade) < max_seq:
                    batch_cascade.append(np.zeros(emb_dim))
            val_step(np.array(batch_val), np.array(batch_val_labels))

        pred = list()
        for i in range(len(test) // batch_size + 1):
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

        template = 'Epoch {:2}, Time: {:.3f}s, Train Loss: {:.3f}, Val Loss: {:.3f}, ' \
                   'Test Loss: {:.3f}, LOG2 MSLE: {:.3f}'
        print(template.format(epoch + 1, time.time() - time_start,
                              train_loss.result(), val_loss.result(), test_loss.result(), report_loss))

        if val_loss.result() < best_val_loss:
            best_val_loss = val_loss.result()
            save_predictions = pred
            patience = FLAGS.patience

            # save model
            student.save_weights(FLAGS.student_path + FLAGS.name + '-student-' + str(FLAGS.num) + '.h5')
            print('Model saved!')

        if patience == 0:
            report_loss = np.mean(np.square(np.log2(np.array([pre if pre >= 1 else 1 for pre in save_predictions])) -
                                            np.log2(np.array([tru if tru >= 1 else 1 for tru in list(test_y)]))))

            print('Predictions saved! Best Test MSLE: {}'.format(report_loss))
            break
        else:
            patience -= 1

    print('Finished! Time used: {:.3f}min'.format((time.time() - start_time) / 60))


if __name__ == '__main__':
    app.run(main)
