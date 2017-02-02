DenseNet with TensorFlow
========================

Already done
------------

Two types of `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`__ (DenseNets) are available:

- DenseNet - without bottleneck layers
- DenseNet-BC - with bottleneck layers

Each model can be tested on such datasets:

- Cifar10
- Cifar10+ (with data augmentation)
- Cifar100
- Cifar100+ (with data augmentation)
- SVHN

A number of layers, blocks, growth rate and other training params may be changed trough shell or inside the source code.

Example run:

.. code::

    python run_dense_net.py --train --test --dataset=C10

List all available options:

.. code:: 
    
    python run_dense_net.py --help

There are also many `other implementations <https://github.com/liuzhuang13/DenseNet>`__ - they may be useful also.

Citation:

.. code::
     
     @article{Huang2016Densely,
            author = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
            title = {Densely Connected Convolutional Networks},
            journal = {arXiv preprint arXiv:1608.06993},
            year = {2016}
     }

Difference compared to the `original <https://github.com/liuzhuang13/DenseNet>`__ implementation
---------------------------------------------------------
The existing model should use identical hyperparameters to the original code. If you note some errors - please open an issue.

Dependencies
------------

- Model was tested with Python 3.4.3+ and Python 3.5.2 with and without CUDA.
- Model should work as expected with TensorFlow >= 0.10.

Repo supported with requirements file - so the easiest way to install all just run ``pip install -r requirements.txt``.

What should be done
-------------------
In future, I hope to make such improvements:

- Add data provider for SVHN dataset
- Add data provider and model for ImageNet

