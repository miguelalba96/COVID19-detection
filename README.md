# COVID19-detection
This is a repository to try new methods
detecting covid 19 from images
|Pneumonia| Covid-19|
:---------------:|:---------------:
![lime_pneumonia](https://github.com/miguelalba96/COVID19-detection/blob/master/images/pneumonia_lime.png) | ![lime_covid](https://github.com/miguelalba96/COVID19-detection/blob/master/images/covid_lime.png)

(*Explanable features*)

## Datasets
Principal COVID-19 dataset is taken on the repository https://github.com/ieee8023/covid-chestxray-dataset, pnuemonia dataset from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

## Preprocessing
The main data preparation and preprocessing is based on the repository [COVID-NET](https://github.com/lindawangg/COVID-Net) including more modularity for new data, and the option to write datasets as tfrecords.
In order to prepare the data clone the repository `git clone https://github.com/ieee8023/covid-chestxray-dataset` (covid19 image dataset) then download and extract the pnuemonia data in a local a folder `xx/COVID`.
* Every time the covid dataset repository gets updated clone it again for new data and start a new preprocessing.
* Run `preprocessing.py --data-folder .xx/COVID --resize-img xxx`, image size in COVID-NET repo is 224, it is an optional parameter here.
* Go to `/trained_models/*modelname*` and run tensorboard in order to see the training curves

## Models 
All models train in this repository use Tensorflow 2.0.1
* `conv_nets` package contains the models and custom layers

## Training and evaluation
* The custom CNN training loop accepts subclassed models, functional and sequential constructions.
* Create your models in `conv_nets/models.py`
* Use `debug()` function as guide to train.
* Evaluation computes classification reports for the trained model
* In order to evaluate a trained model use `python evaluation.py --model-name 'your model name' --data-path './datapath'
`.

## Additionals
* Adversarial examples crafting addded to the repo, augmentation can be performed crafting modifications of the original images with small pixel changes.
* Explainable Machine Learning (TO ADD)
