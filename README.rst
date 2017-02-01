DenseNet with TensorFlow
========================

Already done
------------

Two types of DenseNet(with and without bottleneck layers) `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`__ were implemented in TensorFlow.
Now it can be run with Cifar10/Cifar100 datasets(with and without augmentation) with various numbers of layers/blocks, so you can check any architecture you want and validate results provided in the paper.
There are also many `other implementations <https://github.com/liuzhuang13/DenseNet>`__ - they may be useful also.

Citation:

.. code::
     
     @article{Huang2016Densely,
            author = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
            title = {Densely Connected Convolutional Networks},
            journal = {arXiv preprint arXiv:1608.06993},
            year = {2016}
     }

What should be done
-------------------
In future I hope make such improvements:

- Add data provider for SVHN dataset
- Add data provider and model for ImageNet

Training
--------
To start train default DenseNet model(L=40, k=12) run ``python run_dense_net.py --train``.
If you want to perform testing after train run ``python run_dense_net.py --test``.
Some params can be changed from command line, some - inside starting script.

Difference compared to the `original <https://github.com/liuzhuang13/DenseNet>`__ implementation
---------------------------------------------------------
Existing model should use identical hyperparameters to the original code. If you note some errors - please open an issue.

Dependencies
------------

- Model was tested with Python 3.4.3+ and Python 3.5.2 with and without CUDA.
- Model should work as expected with TensorFlow >= 0.10.

Repo supported with requirements file - so the easiest way to install all just run ``pip install -r requirements.txt``
