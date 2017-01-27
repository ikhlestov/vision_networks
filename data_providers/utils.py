from .cifar import CifarDataProvider


def get_data_provider_by_name(name, train_params):
    """Return required data provider class"""
    shuffle = train_params['shuffle']
    if name in ['C10', 'C10+', 'C100', 'C100+']:
        validate = train_params.get('validate', False)
        normalize = train_params['normalize_data']
        if validate:
            validation_split = train_params['validation_split']
        else:
            validation_split = None
    if name == 'C10':
        return CifarDataProvider(
            cifar_class=10, shuffle=shuffle,
            normalize=normalize,
            validation_split=validation_split)
    if name == 'C10+':
        return CifarDataProvider(
            cifar_class=10, shuffle=shuffle,
            normalize=normalize,
            validation_split=validation_split,
            data_augmentation=True)
    if name == 'C100':
        return CifarDataProvider(
            cifar_class=100, shuffle=shuffle,
            normalize=normalize,
            validation_split=validation_split)
    if name == 'C100+':
        return CifarDataProvider(
            cifar_class=100, shuffle=shuffle,
            normalize=normalize,
            validation_split=validation_split,
            data_augmentation=True)
    else:
        print("Sorry, data provider for `%s` dataset "
              "was not implemented yet" % name)
        exit()
