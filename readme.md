# EE6483 Mini Project: Cats vs Dogs

This is a CV sub-task of mini-project EE6483, the main task is to complete a cat and dog classification task and extend the classifier to CIFAR-10.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [How to perform the experiments](#experiments)
- [Code Structure](#code-structure)

## Installation

1. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage

To run the training and evaluation pipeline, use the following command:
```bash
python run.py --config path/to/config.yaml
```

You can override the configuration parameters using command-line arguments. For example:
```bash
python run.py --config path/to/config.yaml --epochs 20 --batch_size 64
```

## Configuration

The configuration file (`config.yaml`) contains various settings for the dataset, model, training, and evaluation. Below is an example configuration:

```yaml
dataset:
  name: "custom"  # or "CIFAR10"
  augmentations: "resize"
  use_normalize: true
  imgsize: 128
  crop_type: "random"  # or "center"
  batch_size: 32
  use_mixup: true
  use_cutmix: true

model:
  name: "resnet50"
  pretrained: true

training:
  epochs: 10
  optimizer: "Adam"
  lr: 0.001
  lr_decay:
    step_size: 7
    gamma: 0.1
  patience: 3
  seeds: [42, 43, 44]

evaluation:
  num_classes: 10
  class_names: ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
```

One thing to note is that if seed is only entered as a number, it will only be run once with that number as the seed, and if you want to run it multiple times, you will need to set up more than one seed.

## How to perform the experiments
To perform the experiments, follow these steps:

### Data Preprocessing

The data preprocessing steps are defined in `dataloader.py`. The following augmentations and transformations are available:

- **Random Crop**: Randomly crops the image.
- **Center Crop**: Crops the image at the center.
- **Random Horizontal Flip**: Randomly flips the image horizontally.
- **Auto Augment**: Applies automatic augmentation policies.
- **Rand Augment**: Applies random augmentations.
- **Normalization**: Normalizes the image using mean and standard deviation.
- **Mixup**: generates new training samples by linearly mixing two images by weights and weighting the labels equally to improve the generalization ability of the model.
- **Cutmix**: pastes random regions of one image onto another and mixes the labels in proportion to the pasted regions to enhance data diversity and robustness.

### Models

The following models are available and defined in `model.py`:

- **ResNet50**: A 50-layer deep convolutional neural network.
- **ResNet50 (Untrained)**: A 50-layer deep convolutional neural network without pre-trained weights.
- **ViT-B**: Vision Transformer base model.
- **ConvNeXtV2**: A convolutional neural network with a modern architecture.

### To Use
In this assignment, we need to select or build at least one machine learning model to construct the classifier. Clearly describe the model you are using, including the model architecture diagram, input and output dimensions, model structure, loss function, training strategy, etc. If using non-deterministic methods, ensure reproducibility of results by averaging results over multiple runs, cross-validation, fixing random seeds, etc. So, we often need to use a custom config multiple times for multiple model runs and evaluations. Mean + 95% confidence intervals are automatically run multiple times and calculated when you give multiple seeds. Confusion matrices for models are saved with timestamps uniquely named, so please note the order in which your models are used. Model predictions and results will be saved in log, or in the case of using the Cat vs Dogs dataset, the results of the unlabeled test set will be saved in a CSV file (again uniquely named with a timestamp), along with the output of two positive/false verdict images each of the predictions from the validation set.


## Code Structure

- `run.py`: Main script to run the training and evaluation pipeline.
- `dataloader.py`: Contains functions for data loading and augmentation.
- `model.py`: Defines the model architecture and initialization.
- `utils.py`: Utility functions for training, evaluation, and logging.
- `config.yaml`: Configuration file for setting parameters.

