DenseNet with TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~

Two types of `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`__ (DenseNets) are available:

- DenseNet - without bottleneck layers
- DenseNet-BC - with bottleneck layers

Each model can be tested on such datasets:

- Cifar10
- Cifar10+ (with data augmentation)
- Cifar100
- Cifar100+ (with data augmentation)
- SVHN

A number of layers, blocks, growth rate, image normalization and other training params may be changed trough shell or inside the source code.

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

Test run
--------

Test results on various datasets. Image normalization per channels was used. Results reported in paper provided in parenthesis. For Cifar+ datasets image normalization was performed before augmentation. This may cause a little bit lower results than reported in paper.

====================== ====== =========== =========== ============== ==============
Model type             Depth  C10          C10+       C100           C100+
====================== ====== =========== =========== ============== ==============
DenseNet(*k* = 12)     40     6.67(7.00)  5.44(5.24)  27.44(27.55)   25.62(24.42)
DenseNet-BC(*k* = 12)  100    5.54(5.92)  4.87(4.51)  24.88(24.15)   22.85(22.27)
====================== ====== =========== =========== ============== ==============

Approximate training time for models on GeForce GTX TITAN X GM200 (12 GB memory):

- DenseNet(*k* = 12, *d* = 40) - 17 hrs
- DenseNet-BC(*k* = 12, *d* = 100) - 1 day 18 hrs


Difference compared to the `original <https://github.com/liuzhuang13/DenseNet>`__ implementation
---------------------------------------------------------
The existing model should use identical hyperparameters to the original code. If you note some errors - please open an issue.

Also it may be useful to check my blog post `Notes on the implementation DenseNet in tensorflow. <https://medium.com/@illarionkhlestov/notes-on-the-implementation-densenet-in-tensorflow-beeda9dd1504#.55qu3tfqm>`__

Dependencies
------------

- Model was tested with Python 3.4.3+ and Python 3.5.2 with and without CUDA.
- Model should work as expected with TensorFlow >= 0.10. Tensorflow 1.0 support was recently included.

Repo supported with requirements files - so the easiest way to install all just run:

- in case of CPU usage ``pip install -r requirements/cpu.txt``.
- in case of GPU usage ``pip install -r requirements/gpu.txt``.

