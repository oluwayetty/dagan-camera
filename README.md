# DAGAN
Implementation of DAGAN: Data Augmentation Generative Adversarial Networks

## Introduction

This is an implementation of DAGAN as described in https://arxiv.org/abs/1711.04340. The implementation provides data loaders, model builders, model trainers, and synthetic data generators for the Omniglot and VGG-Face datasets.

## Datasets


## Training a DAGAN

After the datasets are downloaded and the dependencies are installed, a DAGAN can be trained by running:

```
python train_camera_dagan.py --batch_size 8 --generator_inner_layers 3 --discriminator_inner_layers 5 --num_generations 64 --experiment_title camera-dagan --num_of_gpus 1 --z_dim 24 --dropout_rate_value 0.5
```

Here, `generator_inner_layers` and `discriminator_inner_layers` refer to the number of inner layers per MultiLayer in the generator and discriminator respectively. `num_generations` refers to the number of samples generated for use in the spherical interpolations at the end of each epoch.

## Multi-GPU Usage

Implementation supports multi-GPU training. Simply pass `--num_of_gpus <x>` to the script to train on  x GPUs (note that this only works if the GPUs are on the same machine).

## Defining a new task for the DAGAN

If you want to train your own DAGAN on a new dataset you need to do the following:

1. Edit data.py and define a new data loader class that inherits from either DAGANDataset or DAGANImblancedDataset. The first class is used when a dataset is balanced (i.e. every class has the same number of samples), the latter is for when this is not the case.

An example class for a balanced dataset is:

```
class CameraDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches):
      super(CameraDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels, num_of_gpus,
                                                gen_batches)

    def load_dataset(self, gan_training_index):

        self.train1 = np.load("datasets/train_cam1_1.npy")

        self.val1 = np.load("datasets/val_cam2_1.npy")

        self.test1 = np.load("datasets/test_cam3_1.npy")

        x_train = self.train1[:1000] / np.max(self.train1[:1000]) #normalizing data
        x_train = np.reshape(x_train, newshape=(1, 1000, 224, 224, 3))

        x_test = self.test1[:1000] / np.max(self.test1[:1000]) #normalizing data
        x_test = np.reshape(x_test, newshape=(1, 1000, 224, 224, 3))

        x_val = self.val1[:1000] / np.max(self.val1[:1000]) #normalizing data
        x_val = np.reshape(x_val, newshape=(1, 1000, 224, 224, 3))

 ```

In short, you need to define your own load_dataset function. This function should load your dataset in the form [num_classes, num_samples, im_height, im_width, im_channels]. Make sure your data values lie within the 0.0 to 1.0 range otherwise the system will fail to model them. Then you need to choose which classes go to each of your training, validation and test sets.

2. Once your data loader is ready, use a template such as train_camera_dagan.py and change the data loader that is being passed. This should be sufficient to run experiments on any new image dataset.

## To Generate Data

The model training automatically uses unseen data to produce generations at the end of each epoch. However, once you have trained a model to satisfication you can generate samples for the whole of the validation set using the following command:

```
python gen_camera_dagan.py -batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 5 --num_generations 64 --experiment_title omniglot_dagan_experiment_default --num_of_gpus 1 --z_dim 100 --dropout_rate_value 0.5 --continue_from_epoch 38
```
All the arguments must match the trained network's arguments and the `continue_from_epoch` argument must correspond to the epoch the trained model was at.

## Acknowledgements

Furthermore, the interpolations used in this project are a result of the <a href="https://arxiv.org/abs/1609.04468" target="_blank">Sampling Generative Networks paper</a> by Tom White.
The code itself was found at https://github.com/dribnet/plat.
