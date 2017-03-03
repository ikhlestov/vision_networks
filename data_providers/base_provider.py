import numpy as np


class DataSet:
    """Class to represent some dataset: train, validation, test"""
    @property
    def num_examples(self):
        """Return qtty of examples in dataset"""
        raise NotImplementedError

    def next_batch(self, batch_size):
        """Return batch of required size of data, labels"""
        raise NotImplementedError


class ImagesDataSet(DataSet):
    """Dataset for images that provide some often used methods"""

    def _measure_mean_and_std(self):
        # for every channel in image
        means = []
        stds = []
        # for every channel in image(assume this is last dimension)
        for ch in range(self.images.shape[-1]):
            means.append(np.mean(self.images[:, :, :, ch]))
            stds.append(np.std(self.images[:, :, :, ch]))
        self._means = means
        self._stds = stds

    @property
    def images_means(self):
        if not hasattr(self, '_means'):
            self._measure_mean_and_std()
        return self._means

    @property
    def images_stds(self):
        if not hasattr(self, '_stds'):
            self._measure_mean_and_std()
        return self._stds

    def shuffle_images_and_labels(self, images, labels):
        rand_indexes = np.random.permutation(images.shape[0])
        shuffled_images = images[rand_indexes]
        shuffled_labels = labels[rand_indexes]
        return shuffled_images, shuffled_labels

    def normalize_images(self, images, normalization_type):
        """
        Args:
            images: numpy 4D array
            normalization_type: `str`, available choices:
                - divide_255
                - divide_256
                - by_chanels
        """
        if normalization_type == 'divide_255':
            images = images / 255
        elif normalization_type == 'divide_256':
            images = images / 256
        elif normalization_type == 'by_chanels':
            images = images.astype('float64')
            # for every channel in image(assume this is last dimension)
            for i in range(images.shape[-1]):
                images[:, :, :, i] = ((images[:, :, :, i] - self.images_means[i]) /
                                       self.images_stds[i])
        else:
            raise Exception("Unknown type of normalization")
        return images

    def normalize_all_images_by_chanels(self, initial_images):
        new_images = np.zeros(initial_images.shape)
        for i in range(initial_images.shape[0]):
            new_images[i] = self.normalize_image_by_chanel(initial_images[i])
        return new_images

    def normalize_image_by_chanel(self, image):
        new_image = np.zeros(image.shape)
        for chanel in range(3):
            mean = np.mean(image[:, :, chanel])
            std = np.std(image[:, :, chanel])
            new_image[:, :, chanel] = (image[:, :, chanel] - mean) / std
        return new_image


class DataProvider:
    @property
    def data_shape(self):
        """Return shape as python list of one data entry"""
        raise NotImplementedError

    @property
    def n_classes(self):
        """Return `int` of num classes"""
        raise NotImplementedError

    def labels_to_one_hot(self, labels):
        """Convert 1D array of labels to one hot representation
        
        Args:
            labels: 1D numpy array
        """
        new_labels = np.zeros((labels.shape[0], self.n_classes))
        new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
        return new_labels

    def labels_from_one_hot(self, labels):
        """Convert 2D array of labels to 1D class based representation
        
        Args:
            labels: 2D numpy array
        """
        return np.argmax(labels, axis=1)
