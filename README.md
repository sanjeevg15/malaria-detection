# Adversarially Robust Model for Malaria Detection

This repository utilizes an ensemble of three Convolutional Neural Networks to predict whether or not a histopathological image of a cell is malarial. The diversity of the individual models provides a layer of defense against adversarial attacks using the FGSM technique. The dataset used for this study is a collection of 27588 segmented cell images acquired at Chittagong Medical College Hospital, Bangladesh. Additional details about the dataset can be found [here](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets)

Usage:

1. Clone the repository : `git clone  https://github.com/sanjeevg15/malaria-detection.git`
2. To train a model, run : `python train.py -root_dir <path to dataset> -model_name <name of model> -epochs <number of epochs> -batch_size <batch size> -lr <learning rate>`
