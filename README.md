# Adversarially Robust Model for Malaria Detection

This repository utilizes an ensemble of three Convolutional Neural Networks to predict whether or not a histopathological image of a cell is malarial. The diversity of the individual models provides a layer of defense against adversarial attacks using the FGSM technique. The dataset used for this study is a collection of 27588 segmented cell images acquired at Chittagong Medical College Hospital, Bangladesh. Additional details about the dataset can be found [here](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets)

## Description

Three models, as described below, were trained on the Malarial Cell Images dataset. The models were then combined into an ensemble to provide a layer of defense against adversarial attacks. The models were trained using the Adam optimizer with a learning rate of 0.001. The models were trained for 10 epochs with a batch size of 32. The models were trained on a Kaggle Notebook GPU.

1. Model 1: A simple 3-layered CNN with shown in Figure 1. This model will be refered to as "MalariaNet" in future references.
2. Model 2: A resnet50 model with the last layer removed and replaced with a fully connected layer with 2 outputs. This model will be refered to as "ResNet50" in future references.
3. Model 3: A VGG16 model with the last layer removed and replaced with a fully connected layer with 2 outputs. This model will be refered to as "VGG16" in future references.

## Experiments & Results

### Experiment 1: Robustness to Noise

We test the robustness of each of the models to random speckle noise. Speckle noise is defined mathematically as:
$$\mathbf{N} = \mathbf{I} + \mathbf{I} \cdot \mathbf{G}$$
where $\mathbf{I}$ is the original image, $\mathbf{G}$ is a matrix of values from a Gaussian distribution, and $\mathbf{N}$ is the noisy image. The noisy image is then passed through each of the models and the models confidence value for class 'Parasitic' is recorded. This is averaged over ~250 images. The results are shown in Figure 2.

![alt text](figures/noise_study1.png)
![alt text](figures/noise_study2.png)

*Figure 2: Model Predictions on a Speckle Noise Corrupted Uninfected Image. y-axis represents the probability of the model predicting the image to be parasitic. The x-axis represents the standard deviation of the Gaussian distribution used to generate the speckle noise*

## Usage

1. Clone the repository : `git clone  https://github.com/sanjeevg15/malaria-detection.git`
2. To train a model, run : `python train.py -root_dir <path to dataset> -model_name <name of model> -epochs <number of epochs> -batch_size <batch size> -lr <learning rate>`
