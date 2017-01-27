class DataSet:
    """Class to represent some dataset: train, validation, test"""
    @property
    def num_examples(self):
        """Return qtty of examples in dataset"""
        raise NotImplementedError

    def next_batch(self, batch_size):
        """Return batch of required size of data, labels"""
        raise NotImplementedError


class DataProvider:
    @property
    def data_shape(self):
        """Return shape as python list of one data entry"""
        raise NotImplementedError

    @property
    def n_classes(self):
        """Return `int` of num classes"""
        raise NotImplementedError
