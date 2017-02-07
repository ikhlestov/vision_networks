import tempfile
import os
import scipy.io

import numpy as np

from .base_provider import ImagesDataSet, DataProvider
from .downloader import download_data_url


class SVHNDataSet(ImagesDataSet):
    n_classes = 10

    def __init__(self, images, labels, shuffle, normalization):
        """
        Args:
            images: 4D numpy array
            labels: 2D or 1D numpy array
            shuffle: `bool`, should shuffle data or not
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
        """
        self.shuffle = shuffle
        self.images = images
        self.labels = labels
        self.normalization = normalization
        self.start_new_epoch()

    def start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle:
            self.images, self.labels = self.shuffle_images_and_labels(
                self.images, self.labels)

    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self, batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        images_slice = self.images[start: end]
        labels_slice = self.labels[start: end]
        # due to memory error it should be done inside batch
        if self.normalization is not None:
            images_slice = self.normalize_images(
                images_slice, self.normalization)
        if images_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice


class SVHNDataProvider(DataProvider):
    def __init__(self, save_path=None, validation_set=None,
                 validation_split=None, shuffle=False, normalization=None,
                 one_hot=True, **kwargs):
        """
        Args:
            save_path: `str`
            validation_set: `bool`.
            validation_split: `int` or None
                float: chunk of `train set` will be marked as `validation set`.
                None: if 'validation set' == True, `validation set` will be
                    copy of `test set`
            shuffle: `bool`, should shuffle data or not
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            one_hot: `bool`, return lasels one hot encoded
        """
        self._save_path = save_path
        train_images = []
        train_labels = []
        for part in ['train', 'extra']:
            images, labels = self.get_images_and_labels(part, one_hot)
            train_images.append(images)
            train_labels.append(labels)
        train_images = np.vstack(train_images)
        if one_hot:
            train_labels = np.vstack(train_labels)
        else:
            train_labels = np.hstack(train_labels)
        if validation_set and validation_split:
            rand_indexes = np.random.permutation(train_images.shape[0])
            valid_indexes = rand_indexes[:validation_split]
            train_indexes = rand_indexes[:validation_split]
            valid_images = train_images[valid_indexes]
            valid_labels = train_labels[valid_indexes]
            train_images = train_images[train_indexes]
            train_labels = train_labels[train_indexes]
            self.validation = SVHNDataSet(
                valid_images, valid_labels, shuffle, normalization)

        self.train = SVHNDataSet(
            train_images, train_labels, shuffle, normalization)

        test_images, test_labels = self.get_images_and_labels('test', one_hot)
        self.test = SVHNDataSet(test_images, test_labels, False, normalization)

        if validation_set and not validation_split:
            self.validation = self.test

    def get_images_and_labels(self, name_part, one_hot=False):
        url = self.data_url + name_part + '_32x32.mat'
        download_data_url(url, self.save_path)
        filename = os.path.join(self.save_path, name_part + '_32x32.mat')
        data = scipy.io.loadmat(filename)
        images = data['X'].transpose(3, 0, 1, 2)
        labels = data['y'].reshape((-1))
        labels[labels == 10] = 0
        if one_hot:
            labels = self.labels_to_one_hot(labels)
        return images, labels

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = os.path.join(tempfile.gettempdir(), 'svhn')
        return self._save_path

    @property
    def data_url(self):
        return "http://ufldl.stanford.edu/housenumbers/"

    @property
    def data_shape(self):
        return (32, 32, 3)


if __name__ == '__main__':
    # WARNING: this test will require about 5 GB of RAM
    import matplotlib.pyplot as plt

    def plot_images_labels(images, labels, axes, main_label):
        plt.text(0, 1.5, main_label, ha='center', va='top',
                 transform=axes[len(axes) // 2].transAxes)
        for image, label, axe in zip(images, labels, axes):
            axe.imshow(image)
            axe.set_title(np.argmax(label))
            axe.set_axis_off()

    n_plots = 10
    fig, axes = plt.subplots(nrows=2, ncols=n_plots)

    dataset = SVHNDataProvider()
    plot_images_labels(
        dataset.train.images[:n_plots],
        dataset.train.labels[:n_plots],
        axes[0],
        'Original dataset')

    dataset = SVHNDataProvider(shuffle=True)
    plot_images_labels(
        dataset.train.images[:n_plots],
        dataset.train.labels[:n_plots],
        axes[1],
        'Shuffled dataset')

    plt.show()
