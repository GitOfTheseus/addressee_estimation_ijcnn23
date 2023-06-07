# A Deep Learning Model for Addressee Estimation in Social Robots

Addressee Estimation in multi-party Human-Robot Interaction

## Brief Intro to the project

This repository holds the code used to train and test a Hybrid Deep Learning model whose results are published in the Conference Paper "To Whom are You Talking? A DL model to Endow Social Robots with Addressee Estimation Skills" presented at the International Joint Conference on Neural Networks (IJCNN) 2023.
Please go to (DOI should yet be released) to read the paper.

# Requirements

Python Version 3.8.10 \
Libraries --> See file requirements.txt to set the correct virtual environment	

## Description

Addressee Estimation is the ability to understand to whom a person is directing an utterance. This ability is crucial for social robot engaging in multi-party interaction to understand the basic dynamics of social communication. \
In this project, we trained a DL model composed of convolutional layers and LSTM cells and taking as input visual information of the speaker to estimate the placement of the addressee. 
We used a supervised learning approach and the data to train our model were taken from the Vernissage Corpus, a dataset collected in multi-party Human-Robot Interaction from the robot's sensors. For the dataset, see http://vernissage.humavips.eu/ \
Specifically, we extract two visual features: body pose vectors and face images of the speaker from the video stream collected with the robot NAO's cameras and we use them to feed our model.

## Installation
You can run the Python code creating a virtual environment and installing the requirements listed in the file requirements.txt

## Support
If you need any support, please write an email to carlo.mazzola@iit.it

## Authors and acknowledgment

Author of the code:
Carlo Mazzola

Acknoweledgement:
Thanks to Marta Romeo (second author of the paper) for her support in designing the architecture and in several steps of the development 

## License
Creative Commons Attribution 4.0 International (CC-BY-4.0)
https://joinup.ec.europa.eu/licence/creative-commons-attribution-40-international-cc-40

For attribution, cite the paper (DOI released after the conference)

## Project status
