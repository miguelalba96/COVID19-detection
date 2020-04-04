import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import utils
from net_tools import DataLoader, Checkpoint, write_tensorboard, build_graph


class CNNCovid19(object):
    def __init__(self, modelname, data_path, model, hyperparams, **kwargs):
        """
        :param modelname:
        :param data_path: data of records
        :param model: tensorflow model
        :param hyperparams: params
        :param kwargs: 
        """
        self.model_path = os.path.join('/.trained_models', modelname)
        self.data_path = data_path
        self.params = {
            'batch_size': 128,
            'learning_rate': 0.001,
            'schedule': False,
            'optimizer': 'ADAM',
            'test_iter': 100,
            'l1l2': 0.0001,
            'epochs': 50,
            'max_class_samples': 8514  # number of pneumonia cases in the data
        }
        self.params.update(hyperparams)
        self.log_dir, self.ckpt_dir, self.train_writer, self.test_writer = self.create_dirs()

        self.steps_epoch = np.ceil(2.0 * self.params['max_class_samples'] / self.params['batch_size'])
        self.epochs = self.params['epochs']
        self.epoch_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
        self.step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

        self.model = model
        self.train_data = DataLoader(self.data_path, training=True)
        self.test_data = DataLoader(self.data_path, training=False)

        self.lr, self.opt = self.optimizer()
        self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        self.train_loss, self.test_loss, self.train_acc, self.test_acc = self.build_metrics()

        architecture = dict(model=self.model,
                            optimizer=self.opt,
                            current_epoch=self.epoch_counter,
                            step=self.step)
        self.ckpt = Checkpoint(architecture, self.ckpt_dir, max_to_keep=3)
        try:
            self.ckpt.restore().assert_existing_objects_matched()
            print('Loading pre trained model')
        except Exception as e:
            print(e)

    def create_dirs(self):
        log_dir = os.path.join(self.model_path, 'logs')
        ckpt_dir = os.path.join(self.model_path, 'checkpoints')
        train_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'opt/train'))
        test_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'opt/test'))
        utils.mdir(self.log_dir)
        utils.mdir(self.ckpt_dir)
        return log_dir, ckpt_dir, train_writer, test_writer

    def build_metrics(self):
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
        test_acc = tf.keras.metrics.CategoricalAccuracy(name='test_acc')
        return train_loss, test_loss, train_acc, test_acc

    def optimizer(self):
        if self.params['schedule']:
            lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.params['learning_rate'],
                                                                decay_steps=8000, decay_rate=0.5, staircase=True)
        else:
            lr = self.params['learning_rate']
        if self.params['optimizer'] == 'SGDM':
            opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif self.params['optimizer'] == 'ADAM':
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            raise NotImplementedError
        return lr, opt

    def cat_cross_entropy(self, data, labels, training):
        predictions = self.model(data, training)
        obj_loss = self.loss(y_true=labels, y_pred=predictions)
        return obj_loss

    @tf.function
    def train_step(self, data, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(data, training=True)
            loss = self.cat_cross_entropy(data, labels, training=True)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        _loss = self.train_loss(loss)
        _acc = self.train_acc(labels, predictions)
        summary = {'type': 'train', 'loss': loss, 'accuracy': _acc}
        return summary

    @tf.function
    def test_step(self, data, labels):
        predictions = self.model(data, training=False)
        loss = self.cat_cross_entropy(data, labels, training=False)
        _loss = self.test_loss(loss)
        _acc = self.test_acc(labels, predictions)
        summary = {'type': 'test', 'loss': loss, 'accuracy': _acc}
        return summary

    def train(self):
        print('Starting Training')
        train = self.train_data.balanced_batch()
        test = self.test_data.balanced_batch()
        data = tf.data.Dataset.zip((train, test))

        epoch_bar = tqdm.tqdm(total=self.epochs, desc='Epoch', position=0)
        for epoch in range(int(self.epochs)):
            self.epoch_counter.assign_add(1)
            step_bar = tqdm.tqdm(total=50, desc='Batch', position=1)
            for train_batch, test_batch in data:
                img, labels = train_batch
                test_img, test_labels = test_batch
                train_summary = self.train_step(img, labels)
                test_summary = self.test_step(test_img, test_labels)
                if int(self.step) == 0:
                    build_graph(self.model, img, self.log_dir, self.step)
                (train_loss, train_acc) = self.train_loss.result(), self.train_acc.result()
                (test_loss, test_acc) = self.test_loss.result(), self.test_acc.result()
                lr = float(self.lr(self.step))
                if int(self.step % self.params['test_iter']) == 0:
                    with self.train_writer.as_default():
                        write_tensorboard(train_summary, step=self.step)
                        tf.summary.scalar('Metrics/Learning_rate', lr.numpy(), step=self.step)
                        self.train_loss.reset_states()
                        self.train_acc.reset_states()
                    with self.test_writer.as_default():
                        write_tensorboard(test_summary, step=self.step)
                        self.test_loss.reset_states()
                        self.test_acc.reset_states()
                self.step.assign_add(1)
                step_bar.update(1)

                if int(self.step % self.steps_epoch) == 0:
                    with self.train_writer.as_default():
                        write_tensorboard(train_summary, step=self.step, full_eval=True)
                    break

            epoch_bar.update(1)
            for _test in self.test_data.test_dataset():
                test_summary = self.test_step(_test[0], _test[1])
            with self.test_writer.as_default():
                write_tensorboard(test_summary, step=self.step, full_eval=True)

            self.train_writer.flush()
            self.test_writer.flush()

            self.ckpt.save(epoch)
            self.model.save(os.path.join(self.model_path, 'epoch_{:05d}.h5'.format(epoch)))
            self.model.save(os.path.join(self.model_path, 'frozen'))
        print('Finished Training')
        return None


if __name__ == '__main__':
    pass