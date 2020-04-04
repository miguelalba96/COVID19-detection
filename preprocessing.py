import os
import argparse
import pydicom as dicom
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import utils


parser = argparse.ArgumentParser(description='Preprocessing')
parser.add_argument('--data-folder', help="Data folder with pneumonia dataset")
parser.add_argument('--covid-path', default='./covid-chestxray-dataset/', help='covid image path')

args = parser.parse_args()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def encode_label(label):
    if label == 'normal':
        return 0
    if label == 'pneumonia':
        return 1
    if label == 'COVID-19':
        return 2


def _process_examples(example_data, filename: str, channels=1):
    """
    :param example_data: takes the list of dictionaries and transform them into Tf records, this is an special format
    of tensorflow data that makes your life easier in tf 1.x and 2.0 saving the data and load it in our training loop
    (WARNING: You have to take care of the encoding of features to not have problems when loading the data, this means
    taking into consideration that images are int or float)
    :param filename: output filename
    :param channels: number of channels of the image (RGB=3), grayscale=!
    :return: None
    """
    with tf.io.TFRecordWriter(filename) as writer:
        for i, ex in enumerate(example_data):
            # define pre augmentation of pre image resizing
            image = ex['image'].astype(np.float32)
            image = image.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(ex['image'].shape[0]),
                'width': _int64_feature(ex['image'].shape[1]),
                'depth': _int64_feature(channels),
                'crop': _bytes_feature(image),
                'label': _int64_feature(encode_label(ex['label']))
            }))
            writer.write(example.SerializeToString())
    return None


class PrepCovid(object):
    def __init__(self, outdir, pneumonia_path, test_persons=None):
        self.outdir = outdir
        self.pneumonia_path = pneumonia_path

        self.test_dict_persons = {
            'pneumonia': ['8', '31'],
            'COVID-19': ['19', '20', '36', '42', '86'],
            'normal': []
        }
        if test_persons is not None:
            self.test_dict_persons.update(test_persons)

        utils.mdir(os.path.join(self.outdir, 'train_data'))

        self.train_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
        self.test_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}

        self.data_counts = {
            'normal': 0,
            'pneumonia': 0,
            'COVID-19': 0
        }
        self.mapping = {
            'COVID-19': 'COVID-19',
            'SARS': 'pneumonia',
            'MERS': 'pneumonia',
            'Streptococcus': 'pneumonia',
            'Normal': 'normal',
            'Lung Opacity': 'pneumonia',
            '1': 'pneumonia'
        }

        self.pneumonias = ["COVID-19", "SARS", "MERS", "ARDS", "Streptococcus"]
        self.pathologies = ["Pneumonia", "Viral Pneumonia", "Bacterial Pneumonia", "No Finding"] + self.pneumonias
        print('pathologies:{}'.format(sorted(self.pathologies)))

    def shard_dataset(self, dataset, num_records=500):
        chunk = len(dataset) // num_records
        parts = [(k * chunk) for k in range(len(dataset)) if (k * chunk) < len(dataset)]
        return chunk, parts

    def save_data(self, dataset, label, dataname, training=True):
        # shards
        _prefix = 'train' if training is True else 'test'
        train_check = 0
        if len(dataset) > 100:
            chunk, parts = self.shard_dataset(dataset)
            for i, j in enumerate(tqdm(parts)):
                shard = dataset[j:(j + chunk)]
                fn = '{}_{}-{}_{:03d}-{:03d}.tfrecord'.format(_prefix, label, dataname, i + 1, 50)
                _process_examples(shard, os.path.join(self.outdir, 'train_data', fn))
                train_check += len(shard)
            print('Number of samples for {} in training: {}'.format(label, train_check))
        else:
            fn = '{}_{}-{}_{:03d}-{:03d}.tfrecord'.format(_prefix, label, dataname, 1, 1)
            _process_examples(dataset, os.path.join(self.outdir, 'train_data', fn))
            print('Small dataset with {} samples'.format(len(dataset)))
        return None

    def prepare_covid_xray(self):
        csv = pd.read_csv(os.path.join(args.covid_path, 'metadata.csv'), nrows=None)
        idx_pa = csv["view"] == "PA"
        csv = csv[idx_pa]

        train = []
        test = []

        labels = {'normal': [], 'pneumonia': [], 'COVID-19': []}
        for index, row in csv.iterrows():
            f = row['finding']
            if f in self.mapping:
                self.data_counts[self.mapping[f]] += 1
                entry = [int(row['patientid']), row['filename'], self.mapping[f]]
                labels[self.mapping[f]].append(entry)

        print('Data distribution from covid-chestxray-dataset:')
        print(self.data_counts)
        for key in labels.keys():
            data_list = labels[key]
            if len(data_list) == 0:
                print('No data for {} in this dataset'.format(key))
                continue

            print('Key: ', key)
            print('Test patients: ', self.test_dict_persons[key])
            # go through all the patients
            for patient in data_list:
                fn = os.path.join(args.covid_path, 'images', patient[1])
                meta = {
                    'dataset': 'covid-chestxray-dataset',
                    'patient_id': patient,
                    'filename': fn,
                    'image': utils.imread(fn),
                    'label': key,
                    'train': 0 if str(patient[0]) in self.test_dict_persons[key] else 1
                }
                # print(0 if patient[0] in self.test_dict_persons[key] else 1)
                # print(patient[0], self.test_dict_persons[key])
                if meta['train'] == 0:
                    test.append(meta)
                    self.test_count[patient[2]] += 1
                else:
                    train.append(meta)
                    self.train_count[patient[2]] += 1

        for lb in labels.keys():
            train_label = [ex for ex in train if ex['label'] == lb]
            test_label = [ex for ex in test if ex['label'] == lb]
            print('saving records for label: {}'.format(lb))
            if len(train_label) != 0:
                self.save_data(train_label, lb, 'chestxray')
            if len(test_label) != 0:
                self.save_data(test_label, lb, 'chestxray', False)

        print('test count: ', self.test_count)
        print('train count: ', self.train_count)

    def prepare_pneumonia_data(self):
        normal_fn = os.path.join(self.pneumonia_path, 'stage_2_detailed_class_info.csv')
        pneu_fn = os.path.join(self.pneumonia_path, 'stage_2_train_labels.csv')
        csv_normal = pd.read_csv(normal_fn, nrows=None)
        csv_pneu = pd.read_csv(pneu_fn, nrows=None)

        labels = {'normal': [], 'pneumonia': []}

        train = []
        test = []

        for index, row in csv_normal.iterrows():
            if row['class'] == 'Normal':
                labels['normal'].append(row['patientId'])

        for index, row in csv_pneu.iterrows():
            if int(row['Target']) == 1:
                labels['pneumonia'].append(row['patientId'])

        for key in labels.keys():
            list_fns = labels[key]
            if len(list_fns) == 0:
                continue

            # fixed evaluation
            test_patients = np.load(os.path.join(self.pneumonia_path, 'rsna_test_patients_{}.npy'.format(key)))

            for patient in list_fns:
                fn = os.path.join(self.pneumonia_path, 'stage_2_train_images', patient + '.dcm')
                ds = dicom.dcmread(fn)
                meta = {
                    'dataset': 'neumonia_kaggle_Dataset',
                    'id': patient,
                    'filename': fn,
                    'image': ds.pixel_array,
                    'label': key,
                    'train': 1 if patient[0] not in test_patients else 0
                }

                if patient in test_patients:
                    test.append(meta)
                    self.test_count[key] += 1
                else:
                    train.append(meta)
                    self.train_count[key] += 1

            print('{} done!'.format(key))

        for lb in labels.keys():
            train_label = [ex for ex in train if ex['label'] == lb]
            test_label = [ex for ex in test if ex['label'] == lb]
            print('saving records for label: {}'.format(lb))
            if len(train_label) != 0:
                self.save_data(train_label, lb, 'kaggle')
            if len(test_label) != 0:
                self.save_data(test_label, lb, 'kaggle', training=False)

        print('test count: ', self.test_count)
        print('train count: ', self.train_count)

    def prepate_datasets(self):
        """
        add a new function to the class every time preparing a new dataset and call it here
        """

        self.prepare_covid_xray()
        self.prepare_pneumonia_data()

        print('Preprocessed datasets ')
        return None


if __name__ == '__main__':
    prep = PrepCovid(os.path.join(args.data_folder, 'all_images'), args.data_folder)
    prep.prepate_datasets()
    pass