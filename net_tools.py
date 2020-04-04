import os
import glob
import tensorflow as tf


class DataLoader(object):
    def __init__(self, data_path, training):
        self.data_path = data_path
        self.training = training
        self.classes = ['normal', 'pneumonia', 'COVID-19']
        self.seed = 1
        if self.training:
            self.batch_size = 128
            self.buffer = 17000
        else:
            self.batch_size = 64
            self.batch_size = 220

    def parse_record(self, record):
        features = {
            'image': tf.io.FixedLenFeature([], dtype=tf.string),
            'height': tf.io.FixedLenFeature([], dtype=tf.int64),
            'width': tf.io.FixedLenFeature([], dtype=tf.int64),
            'label': tf.io.FixedLenFeature([], dtype=tf.int64),
        }
        record = tf.io.parse_single_example(record, features)
        img = tf.io.decode_raw(record['image'], tf.float32)
        img = tf.reshape(img, [record['height'], record['width'], 1])
        label = tf.one_hot(record['label'], len(self.classes), dtype=tf.float32)
        return img, label

    def load_dataset(self, label):
        files = os.path.join(self.data_path, '{}_{}*.tfrecord'.format(self.training, label))
        filenames = glob.glob(files)
        dataset = tf.data.Dataset.list_files(files, shuffle=True, seed=self.seed)
        dataset = dataset.interleave(lambda fn: tf.data.TFRecordDataset(fn), cycle_length=len(filenames),
                                     num_parallel_calls=min(len(filenames), tf.data.experimental.AUTOTUNE))
        dataset = dataset.map(self.parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(self.buffer, seed=self.seed)
        dataset = dataset.repeat()
        return dataset

    def balanced_batch(self):
        datasets = []
        for cl in self.classes:
            datasets.append(self.load_dataset(cl))
        importance = [0.33, 0.33, 0.33]
        sampled_dataset = tf.data.experimental.sample_from_datasets(datasets, weights=importance)
        sampled_dataset = sampled_dataset.batch(self.batch_size)
        sampled_dataset = sampled_dataset.prefetch(2)
        return sampled_dataset

    def test_dataset(self):
        files = os.path.join(self.data_path, 'test_*.tfrecord')
        filenames = glob.glob(files)
        dataset = tf.data.Dataset.list_files(files, shuffle=True, seed=self.seed)
        dataset = dataset.interleave(lambda fn: tf.data.TFRecordDataset(fn), cycle_length=len(filenames),
                                     num_parallel_calls=min(len(filenames), tf.data.experimental.AUTOTUNE))
        dataset = dataset.map(self.parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat(1)
        dataset = dataset.batch(self.batch_size)
        return dataset


class Checkpoint:
    def __init__(self, checkpoint_kwargs, out_dir, max_to_keep=5, keep_checkpoint_every_n_hours=None):
        self.checkpoint = tf.train.Checkpoint(**checkpoint_kwargs)
        self.manager = tf.train.CheckpointManager(self.checkpoint, out_dir, max_to_keep, keep_checkpoint_every_n_hours)

    def restore(self, save_path=None):
        save_path = self.manager.latest_checkpoint if save_path is None else save_path
        return self.checkpoint.restore(save_path)

    def save(self, file_prefix_or_checkpoint_number=None, session=None):
        if isinstance(file_prefix_or_checkpoint_number, str):
            return self.checkpoint.save(file_prefix_or_checkpoint_number, session=session)
        else:
            return self.manager.save(checkpoint_number=file_prefix_or_checkpoint_number)

    def __getattr__(self, attr):
        if hasattr(self.checkpoint, attr):
            return getattr(self.checkpoint, attr)
        elif hasattr(self.manager, attr):
            return getattr(self.manager, attr)
        else:
            self.__getattribute__(attr)


def write_tensorboard(stats_dict, step, full_eval=False):
    name = 'Epoch metrics' if full_eval else 'Metrics'
    type = stats_dict['type']
    for scope, metric in stats_dict.items():
        if scope == 'loss':
            tf.summary.scalar('{}/Loss'.format(name), metric.numpy(), step)
        if scope == 'accuracy':
            tf.summary.scalar('{}/Accuracy'.format(name), metric.numpy(), step)


def build_graph(model, feats, log_dir, step=0):
    @tf.function
    def tracing(feats):
        pred = model(feats)
        return pred
    writer = tf.summary.create_file_writer(os.path.join(log_dir, 'model_graph'))
    tf.summary.trace_on(graph=True)
    _ = tracing(feats)
    with writer.as_default:
        tf.summary.trace_export(name="graphs", step=step)