# COVID19-detection
This is a repository to try new methods detecting covid 19 from images

## Datasets
Principal COVID-19 dataset is taken on the repository https://github.com/ieee8023/covid-chestxray-dataset, pnuemonia dataset from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

## Preprocessing
The main data preparation and preprocessing is based on the repository [COVID-NET](https://github.com/lindawangg/COVID-Net) including more modularity for new data, and the option to write datasets as tfrecords.
In order to prepare the data clone the repository `git clone https://github.com/ieee8023/covid-chestxray-dataset` (covid19 image dataset) then download and extract the pnumonia data in a local a folder `xx/COVID`.
* Every time the covid dataset repository gets updated clone it again for new data and start a new preprocessing.
* Run `preprocessing.py --data-folder .xx/COVID --resize-img xxx`, image size in COVID-NET repo is 224, it is an optional parameter here.

## Models 
All models train in this repository use Tensorflow 2.0.1
* `conv_nets` package contains the models and custom layers

## Training and evaluation
* The custom CNN training loop accepts subclassed models, functional and sequential constructions.
* Evaulation computes classification reports for the trained model

## Additionals
* Adversarial examples crafting addded to the repo, augmentation can be performed crafting modifications of the original images with small pixel changes.
* Explanable Machine Learning (TO ADD)
