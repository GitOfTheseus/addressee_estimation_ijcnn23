# A Deep Learning Model for Addressee Estimation in Social Robots

Addressee Estimation in multi-party Human-Robot Interaction

## Brief Intro to the project

This repository holds the code used to train and test a Hybrid Deep Learning model whose results are published in the Conference Paper "To Whom are You Talking? A DL model to Endow Social Robots with Addressee Estimation Skills" presented at the International Joint Conference on Neural Networks (IJCNN) 2023.
Please go to (DOI should yet be released) to read the paper.

# Requirements

Python Version 3.8.10 \
Libraries --> See file requirements_python3_8.txt to set the correct virtual environment	

# Description

Addressee Estimation is the ability to understand to whom a person is directing an utterance. This ability is crucial for social robot engaging in multi-party interaction to understand the basic dynamics of social communication. \
In this project, we trained a DL model composed of convolutional layers and LSTM cells and taking as input visual information of the speaker to estimate the placement of the addressee. 
We used a supervised learning approach and the data to train our model were taken from the Vernissage Corpus, a dataset collected in multi-party Human-Robot Interaction from the robot's sensors. For the dataset, see http://vernissage.humavips.eu/ \
Specifically, we extract two visual features: body pose vectors and face images of the speaker from the video stream collected with the robot NAO's cameras and we use them to feed our model.

## Code

The code is divided in three main folders. Two of them have already been released (B_TRAIN and C_TEST_AND_PLOT). A third one (A_CREATE_DATASET) will be released soon. \

Codes in A_FEATURES_EXTRACTION manage the dataset creation phase. For these files, please read the README in A_FEATURES_EXTRACTION folder.

Codes in B_TRAIN manage the training
We followed a 10-fold cross validation approach. The code main_training.py manages all the training phase 
## Installation
You can run the Python code creating a virtual environment and installing the requirements listed in the file requirements.txt

## Support
If you need any support, please write an email to carlo.mazzola@iit.it

## Authors and acknowledgment

Author of the code:
Carlo Mazzola https://orcid.org/0000-0002-9282-9873

Acknoweledgement:
Thanks to Marta Romeo https://orcid.org/0000-0003-4438-0255 (second author of the paper) for her support in designing the architecture and in several steps of the development 

## License
Creative Commons Attribution 4.0 International (CC-BY-4.0)
https://joinup.ec.europa.eu/licence/creative-commons-attribution-40-international-cc-40

For attribution, cite the paper:
C. Mazzola, M. Romeo, F. Rea, A. Sciutti and A. Cangelosi, "To Whom are You Talking? A Deep Learning Model to Endow Social Robots with Addressee Estimation Skills," 2023 International Joint Conference on Neural Networks (IJCNN), Gold Coast, Australia, 2023, pp. 1-10, 
https://doi.org/10.1109/IJCNN54540.2023.10191452

## Project status
