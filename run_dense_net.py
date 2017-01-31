import argparse

from models.dense_net import DenseNet
from data_providers.utils import get_data_provider_by_name

train_params_cifar = {
    'batch_size': 64,
    'n_epochs': 300,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,  # epochs * 0.5
    'reduce_lr_epoch_2': 225,  # epochs * 0.75
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_channels',  # None, divide_256, divide_255, by_channels
}

train_params_svhn = {
    'batch_size': 64,
    'n_epochs': 40,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 20,
    'reduce_lr_epoch_2': 30,
}


def get_train_params_by_name(name):
    if name in ['C10', 'C10+', 'C100', 'C100+']:
        return train_params_cifar
    if name == 'SVHN':
        return train_params_svhn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action='store_true',
        help='Train the model')
    parser.add_argument(
        '--test', action='store_true',
        help='Test model for required dataset if exists')
    parser.add_argument(
        '--model_type', '-m', type=str, choices=['DenseNet', 'DenseNet-BC'],
        default='DenseNet',
        help='What type of model to use')
    parser.add_argument(
        '--growth_rate', '-k', type=int, choices=[12, 24, 40],
        default=12,
        help='Grows rate for every layer, '
             'choices were restricted to used in paper')
    parser.add_argument(
        '--depth', '-d', type=int, choices=[40, 100, 190, 250],
        default=40,
        help='Depth of whole network, restricted to paper choices')
    parser.add_argument(
        '--dataset', '-ds', type=str,
        choices=['C10', 'C10+', 'C100', 'C100+', 'SVHN'],
        default='C10',
        help='What dataset should be used')
    args = parser.parse_args()

    # some params used by default and can be changed inside code
    if args.dataset in ['C10', 'C100', 'SVHN']:
        keep_prob = 0.8
    else:
        keep_prob = 1.0

    default_params = {
        'weight_decay': 1e-4,
        'nesterov_momentum': 0.9,
        'keep_prob': keep_prob,
        # first output - a little bit larger than growth rate
        # maybe should be changed for another archs
        'first_output_features': int(args.growth_rate * 1.35),
        'total_blocks': 3,
        'should_save_logs': True,
        'should_save_model': True,
        'renew_logs_saves': True,
    }
    default_params.update(vars(args))

    # another params dataset/architecture related
    train_params = get_train_params_by_name(args.dataset)
    print("Params:")
    for k, v in default_params.items():
        print("\t%s: %s" % (k, v))
    print("Train params:")
    for k, v in train_params.items():
        print("\t%s: %s" % (k, v))

    if not args.train and not args.test:
        print("You should train or test your network. Please check params.")
        exit()

    print("Prepare training data...")
    data_provider = get_data_provider_by_name(args.dataset, train_params)
    print("Initialize the model..")
    model = DenseNet(data_provider=data_provider, **default_params)
    if args.train:
        print("Data provider train images: ", data_provider.train.num_examples)
        model.train_all_epochs(train_params)
    if args.test:
        if not args.train:
            model.load_model()
        print("Data provider test images: ", data_provider.test.num_examples)
        print("Testing...")
        loss, accuracy = model.test(data_provider.test, batch_size=200)
        print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))
