## EEG-Vision Transformer (EEGViT)

## Overview
EEGViT is a hybrid Vision Transformer (ViT) incorporated with Depthwise Convolution in patch embedding layers. This work is based on 
Dosovitskiy, et al.'s ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929). After finetuning EEGViT pretrained on ImageNet, it achieves a considerable improvement over the SOTA on the Absolute Position task in EEGEyeNet dataset.

This repository consists of four models: ViT pretrained and non-pretrained; EEGViT pretrained and non-pretrained. The pretrained weights of ViT layers are loaded from [huggingface.co](https://huggingface.co/docs/transformers/model_doc/vit).

## Dataset download
Download data for EEGEyeNet absolute position task
```bash
wget -O "./dataset/Position_task_with_dots_synchronised_min.npz" "https://osf.io/download/ge87t/"
```
For more details about EEGEyeNet dataset, please refer to ["EEGEyeNet: a Simultaneous Electroencephalography and Eye-tracking Dataset and Benchmark for Eye Movement Prediction"](https://arxiv.org/abs/2111.05100) and [OSF repository](https://osf.io/ktv7m/)

## Installation

### Requirements

First install the general_requirements.txt

```bash
pip3 install -r general_requirements.txt 
```

### Pytorch Requirements

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

For other installation details and different cuda versions, visit [pytorch.org](https://pytorch.org/get-started/locally/).

## Configuration

The model configuration takes place in hyperparameters.py. The training configuration is contained in config.py.

### config.py

We start by explaining the settings that can be made for running the benchmark:

Choose the task to run in the benchmark, e.g.

```bash
config['task'] = 'LR_task'
```

For some tasks we offer data from multiple paradigms. Choose the dataset used for the task, e.g.

```bash
config['dataset'] = 'antisaccade'
```

Choose the preprocessing variant, e.g.

```bash
config['preprocessing'] = 'min'
```

Choose data preprocessed with Hilbert transformation. Set to True for the standard ML models:

```bash
config['feature_extraction'] = True
```

Include our standard ML models into the benchmark run:

```bash
config['include_ML_models'] = True 
```

Include our deep learning models into the benchmark run:

```bash
config['include_DL_models'] = True
```

Include your own models as specified in hyperparameters.py. For instructions on how to create your own custom models see further below.

```bash
config['include_your_models'] = True
```

Include dummy models for comparison into the benchmark run:

```bash
config['include_dummy_models'] = True
```

You can either choose to train models or use existing ones in /run/ and perform inference with them. Set

```bash
config['retrain'] = True 
config['save_models'] = True 
```

to train your specified models. Set both to False if you want to load existing models and perform inference. 
In this case specify the path to your existing model directory under

```bash
config['load_experiment_dir'] = path/to/your/model 
```

In the model configuration section you can specify which framework you want to use. You can run our deep learning models in both pytorch and tensorflow. Just specify it in config.py, make sure you set up the environment as explained above and everything specific to the framework will be handled in the background.

config.py also allows to configure hyperparameters such as the learning rate, and enable early stopping of models.

### hyperparameters.py

Here we define our models. Standard ML models and deep learning models are configured in a dictionary which contains the object of the model and hyperparameters that are passed when the object is instantiated.

You can add your own models in the your_models dictionary. Specify the models for each task separately. Make sure to enable all the models that you want to run in config.py.

## Running the benchmark

Create a /runs directory to save files while running models on the benchmark.

### benchmark.py

In benchmark.py we load all models specified in hyperparameters.py. Each model is fitted and then evaluated with the scoring function corresponding to the task that is benchmarked.

### main.py

To start the benchmark, run

```bash
python3 main.py
```

A directory of the current run is created, containing a training log, saving console output and model checkpoints of all runs.

## Add Custom Models

To benchmark models we use a common interface we call trainer. A trainer is an object that implements the following methods:

```bash
fit() 
predict() 
save() 
load() 
```

### Implementation of custom models

To implement your own custom model make sure that you create a class that implements the above methods. If you use library models, make sure to wrap them into a class that implements above interface used in our benchmark.

### Adding custom models to our benchmark pipeline

In hyperparameters.py add your custom models into the your_models dictionary. You can add objects that implement the above interface. Make sure to enable your custom models in config.py.
