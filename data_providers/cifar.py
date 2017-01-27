import os
import pickle

import numpy as np


from .base_provider import DataSet, DataProvider
from .downloader import download_data_url


def read_cifar(filenames, cifar_classnum):
    assert cifar_classnum == 10 or cifar_classnum == 100
    if cifar_classnum == 10:
        labels_key = b'labels'
    elif cifar_classnum == 100:
        labels_key = b'fine_labels'

    images_res = []
    labels_res = []
    for fname in filenames:
        with open(fname, 'rb') as f:
            images_and_labels = pickle.load(f, encoding='bytes')
        images = images_and_labels[b'data']
        images = images.reshape(-1, 3, 32, 32).swapaxes(1, 3).swapaxes(1, 2)
        # import ipdb; ipdb.set_trace()
        # images = np.transpose(images, [0, 2, 3, 1])
        images_res.append(images)
        labels_res.append(images_and_labels[labels_key])
    images_res = np.vstack(images_res)
    labels_res = np.hstack(labels_res)
    return images_res, labels_res


class CifarDataSet(DataSet):
    def __init__(self, images, labels, n_classes, shuffle):
        self.images = images
        self.n_classes = n_classes
        self.labels = labels
        self.shuffle = shuffle
        self.start_new_epoch()

    def start_new_epoch(self):
        # renew batch counter
        self._batch_counter = 0
        # perform shuffling if required
        if self.shuffle:
            rand_indexes = np.random.permutation(self.images.shape[0])
            self.images = self.images[rand_indexes]
            self.labels = self.labels[rand_indexes]

    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self, batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        images_slice = self.images[start: end]
        labels_slice = self.labels[start: end]
        if images_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice


class CifarDataProvider(DataProvider):
    def __init__(self,
                 data_augmentation=False,
                 normalize=True,
                 one_hot=True,
                 shuffle=True,
                 save_path=None,
                 validation_split=0.1,
                 cifar_class=10):
        self._n_classes = cifar_class
        if save_path is None:
            save_path = '/tmp/cifar%d' % cifar_class
        data_url = 'http://www.cs.toronto.edu/~kriz/cifar-%d-python.tar.gz' % cifar_class
        download_data_url(data_url, save_path)

        if cifar_class == 10:
            save_path = os.path.join(save_path, 'cifar-10-batches-py')
            train_filenames = [
                os.path.join(
                    save_path,
                    'data_batch_%d' % i) for i in range(1, 6)]
            test_filenames = [os.path.join(save_path, 'test_batch')]

        if cifar_class == 100:
            save_path = os.path.join(save_path, 'cifar-100-python')
            train_filenames = [os.path.join(save_path, 'train')]
            test_filenames = [os.path.join(save_path, 'test')]

        f_names_per_dataset = {
            'train': train_filenames,
            'test': test_filenames,
        }

        for dataset_name, f_names_list in f_names_per_dataset.items():
            images, labels = read_cifar(f_names_list, cifar_class)
            # convert labels to one hot
            if one_hot:
                tmp_labels = np.zeros((labels.shape[0], cifar_class))
                tmp_labels[range(labels.shape[0]), labels] = labels
                labels = tmp_labels
            # augment the data
            if data_augmentation:
                raise NotImplementedError
            # normalize data
            if normalize:
                # normalize per channel
                for channel in range(3):
                    images[:, :, :, channel] = (images[:, :, :, channel] - np.mean(images[:, :, :, channel])) / np.std(images[:, :, :, channel])
            else:
                images = images / 255
            if validation_split and dataset_name == 'train':
                split_idx = int(images.shape[0] * (1 - validation_split))
                train_images = images[:split_idx]
                train_labels = labels[:split_idx]
                train_dataset = CifarDataSet(
                    images=train_images, labels=train_labels,
                    n_classes=cifar_class, shuffle=shuffle)
                setattr(self, 'train', train_dataset)
                valid_images = images[split_idx:]
                valid_labels = labels[split_idx:]
                valid_dataset = CifarDataSet(
                    images=valid_images, labels=valid_labels,
                    n_classes=cifar_class, shuffle=shuffle)
                setattr(self, 'validation', valid_dataset)
            else:
                dataset = CifarDataSet(
                    images=images, labels=labels,
                    n_classes=cifar_class, shuffle=shuffle)
                setattr(self, dataset_name, dataset)

    @property
    def data_shape(self):
        return (32, 32, 3)

    @property
    def n_classes(self):
        return self._n_classes
