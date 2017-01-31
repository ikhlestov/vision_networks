import os
import time
import shutil
from datetime import timedelta

import numpy as np
import tensorflow as tf
import tflearn


class DenseNet:
    def __init__(self, data_provider, growth_rate, depth,
                 first_output_features, total_blocks, keep_prob,
                 weight_decay, nesterov_momentum, model_type, dataset,
                 should_save_logs, should_save_model):
        """
        Class to implement networks from this paper
        https://arxiv.org/pdf/1611.05552.pdf

        Args:
            data_provider: Class, that have all required data sets
            growth_rate: `int`, variable from paper
            depth: `int`, variable from paper
            first_output_features: `int`, how many features will be received
                after first convolution. Should be by some way larger than
                growth_rate
            total_blocks: `int`, paper value == 3
            keep_prob: `float`, keep probability for dropout. If keep_prob = 1
                dropout will be disables
            weight_decay: `float`, weight decay for L2 loss, paper = 1e-4
            nesterov_momentum: `float`, momentum for Nesterov optimizer
            model_type: `str`, 'DenseNet' or 'DenseNet-BC'. Should model use
                bottle neck connections or not.
            dataset: `str`, dataset name
            should_save_logs: `bool`, should logs be saved or not
            should_save_model: `bool`, should model be saved or not
        """
        self.data_provider = data_provider
        self.data_shape = data_provider.data_shape
        self.n_classes = data_provider.n_classes
        self.depth = depth
        self.growth_rate = growth_rate
        self.first_output_features = first_output_features
        self.total_blocks = total_blocks
        self.layers_per_block = (depth - 4) // total_blocks
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        self.dataset_name = dataset
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.batches_step = 0

        self._define_inputs()
        self._build_graph()
        self._initialize_session()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf_ver = int(tf.__version__.split('.')[1])
        if tf_ver <= 10:
            self.sess.run(tf.initialize_all_variables())
        else:
            self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.summary_writer = tf.train.SummaryWriter(self.logs_path)

    @property
    def save_path(self):
        save_path = 'saves/%s' % self.model_identifier
        shutil.rmtree(save_path, ignore_errors=True)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, 'model.chkpt')
        return save_path

    @property
    def logs_path(self):
        logs_path = 'logs/%s' % self.model_identifier
        shutil.rmtree(logs_path, ignore_errors=True)
        os.makedirs(logs_path, exist_ok=True)
        return logs_path

    @property
    def model_identifier(self):
        return "{}_growth_rate={}_depth={}_dataset_{}".format(
            self.model_type, self.growth_rate, self.depth, self.dataset_name)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        self.saver.restore(self.sess, self.save_path)

    def log_loss_accuracy(self, loss, accuracy, epoch, prefix,
                          should_print=True):
        if should_print:
            print("mean loss: %f, mean accuracy: %f" % (loss, accuracy))
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(
                tag='accuracy_%s' % prefix, simple_value=float(accuracy))
        ])
        self.summary_writer.add_summary(summary, epoch)

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='labels')
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')

    def composite_function(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        # BN
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        # convolution
        output = self.conv2d(
            output, out_features=out_features, kernel_size=kernel_size)
        # dropout(in case of training and in case it is no 1.0)
        output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        comp_out = self.composite_function(
            _input, out_features=growth_rate, kernel_size=3)
        # concatenate _input with out from composite function
        output = tf.concat(3, (_input, comp_out))
        return output

    def add_block(self, _input, growth_rate, layers_per_block):
        """Add N H_l internal layers"""
        output = _input
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # call composite function with 1x1 kernel
        out_features = int(_input.get_shape()[-1])
        output = self.composite_function(
            _input, out_features=out_features, kernel_size=1)
        # run average pooling
        output = self.avg_pool(output, k=2)
        return output

    def trainsition_layer_to_classes(self, _input):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, k=last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W = self.weight_variable_xavier(
            [features_total, self.n_classes], name='W')
        bias = self.bias_variable([self.n_classes])
        logits = tf.matmul(output, W) + bias
        return logits

    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

    def avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):
        is_training = tflearn.get_training_mode()
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=is_training,
            updates_collections=None)
        return output

    def dropout(self, _input):
        is_training = tflearn.get_training_mode()
        if self.keep_prob < 1:
            output = tf.cond(
                is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def _build_graph(self):
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block
        # first - initial 3 x 3 conv to first_output_features
        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                self.images,
                out_features=self.first_output_features,
                kernel_size=3)

        # add N required blocks
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(output, growth_rate, layers_per_block)
            # last block exist without transition layer
            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output)

        with tf.variable_scope("Transition_to_classes"):
            logits = self.trainsition_layer_to_classes(output)
        prediction = tf.nn.softmax(logits)

        # Losses
        cross_enthropy = -tf.reduce_mean(
            self.labels * tf.log(prediction + 1e-12))
        self.loss = cross_enthropy
        l2_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # optimizer and train step
        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.train_step = optimizer.minimize(
           cross_enthropy + l2_loss * self.weight_decay)

        correct_prediction = tf.equal(
            tf.argmax(prediction, 1),
            tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        for epoch in range(1, n_epochs + 1):
            print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
            start_time = time.time()
            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                learning_rate = learning_rate / 10
                print("Decrease learning rate, new lr = %f" % learning_rate)

            print("Training...")
            loss, acc = self.train_one_epoch(
                self.data_provider.train, batch_size, learning_rate)
            if self.should_save_logs:
                self.log_loss_accuracy(loss, acc, epoch, prefix='train')

            if train_params.get('validation_set', False):
                print("Validation...")
                loss, acc = self.test(
                    self.data_provider.validation, batch_size)
                if self.should_save_logs:
                    self.log_loss_accuracy(loss, acc, epoch, prefix='valid')

            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch - 2) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if self.should_save_model:
                self.save_model()

    def train_one_epoch(self, data, batch_size, learning_rate):
        tflearn.is_training(True, session=self.sess)
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            images, labels = batch
            feed_dict = {
                self.images: images,
                self.labels: labels,
                self.learning_rate: learning_rate
            }
            fetches = [self.train_step, self.loss, self.accuracy]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, accuracy = result
            total_loss.append(loss)
            total_accuracy.append(accuracy)
            if i % 100 == 0:
                print("Batch %d, loss: %f, accuracy: %f" % (i, loss, accuracy))
            if self.should_save_logs:
                self.batches_step += 1
                self.log_loss_accuracy(
                    loss, accuracy, self.batches_step, prefix='per_batch',
                    should_print=False)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(accuracy)
        return mean_loss, mean_accuracy

    def test(self, data, batch_size):
        tflearn.is_training(False, session=self.sess)
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            feed_dict = {
                self.images: batch[0],
                self.labels: batch[1]
            }
            fetches = [self.loss, self.accuracy]
            loss, accuracy = self.sess.run(fetches, feed_dict=feed_dict)
            total_loss.append(loss)
            total_accuracy.append(accuracy)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(accuracy)
        return mean_loss, mean_accuracy
