# A Deep Learning Model for Addressee Estimation in Social Robots

Addressee Estimation in multi-party Human-Robot Interaction

## Brief Intro to the project

This repository holds the code used to train and test a Hybrid Deep Learning model whose results are published in the Conference Paper "To Whom are You Talking? A DL model to Endow Social Robots with Addressee Estimation Skills" presented at the International Joint Conference on Neural Networks (IJCNN) 2023.
Please go to https://doi.org/10.48550/arXiv.2308.10757 (OA version of https://doi.org/10.1109/IJCNN54540.2023.10191452) to read the paper.

# Requirements

Python Version 3.8.10 for codes in B_TRAIN and C_TEST_AND_PLOT
Python Version 3.10 for codes in A_FEATURES_EXTRACTION

# Description

Addressee Estimation is the ability to understand to whom a person is directing an utterance. This ability is crucial for social robot engaging in multi-party interaction to understand the basic dynamics of social communication. \
In this project, we trained a DL model composed of convolutional layers and LSTM cells and taking as input visual information of the speaker to estimate the placement of the addressee. 
We used a supervised learning approach and the data to train our model were taken from the Vernissage Corpus, a dataset collected in multi-party Human-Robot Interaction from the robot's sensors. For the dataset, see http://vernissage.humavips.eu/ 
Specifically, we extract two visual features: body pose vectors and face images of the speaker from the video stream collected with the robot NAO's cameras and we use them to feed our model.

# Code

The code is divided in three main folders: A_FEATURES_EXTRACTION, B_TRAIN and C_TEST_AND_PLOT.

Codes in A_FEATURES_EXTRACTION manage the dataset creation phase. For these files, please read the README in A_FEATURES_EXTRACTION folder.

Codes in B_TRAIN manage the training. For these files, please read the README in B_TRAIN folder.

Codes in C_TEST_AND_PLOT are used to test the trained model with a test set. In this case, a test set taken from the dataset created after features extraction of the Vernissage Dataset.

# Cloning and Installation


## Cloning the Repository
Before you begin, ensure that you have the following:
- Git installed on your local machine. You can download and install Git from https://git-scm.com/downloads ).
To clone this repository from GitLab, follow these steps:

1. Open your terminal or command prompt.
2. Navigate to the directory where you want to clone the repository. 
3. Write
    ```
    git clone https://gitlab.iit.it/cognitiveInteraction/addressee_estimation_ijcnn23.git 
    ```
   and press Enter to execute the command. Git will clone the repository to your local machine.
4. Once the cloning process is complete, you will have a local copy of the repository in the directory you specified.

## Setting Up the Virtual Environment and Installing Dependencies

Before you begin, ensure that you have Python installed. A_FEATURES_EXTRACTION have been programmed using Python 3.10 while B_TRAIN and C_TEST_AND_PLOT Python 3.8.

To install the virtual environment:
1. Open your terminal or command prompt.
2. Navigate to the directory where you want to create the virtual environment. For ease, it can be the directory where you cloned the project.
3. Run the following command to create a new virtual environment:
    ```
    python3 -m venv myenv
    ```
   Replace `myenv` with the name you want to give to your virtual environment. This command will create a new directory named `myenv` containing the virtual environment.

4. Activate the virtual environment by running the appropriate command for your operating system:
    - On Windows:
        ```
        myenv\Scripts\activate
        ```
    - On macOS and Linux:
        ```
        source myenv/bin/activate
        ```
   After activation, you should see the name of the virtual environment in your command prompt.

5. Once the virtual environment is activated, ensure that you are in the directory containing your requirements file (`requirements.txt`).
6. Run the following command to install dependencies listed in the requirements file:
    ```
    pip install -r requirements.txt
    ```
   Replace `requirements.txt` with the actual name of your requirements file:
   To run A_FEATURES_EXTRACTION codes use requirements_python3_10.txt
   To run B_TRAIN and C_TEST_AND_PLOT use requirements_python3_8.txt

## Support
If you need any support, please write an email to carlo.mazzola@iit.it https://orcid.org/0000-0002-9282-9873

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

Published
