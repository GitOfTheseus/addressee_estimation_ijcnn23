# B_TRAIN

Folder "B_TRAIN" contains codes to train the Addressee Estimation deep learning models on data derived from the features extraction applied to the Vernissage Corpus.

For the training, we followed a 10-fold cross validation approach. The code main_training.py manages all the training phase. 

Python version & requirements 3.8, requirements_python3_8.txt 

# main_training.py

Brief description:
This code train one or more models according to the information provided in input.
here below the possible inputs for the model are explained:

--model: type=str, default='cnn_lstm_mm' : it refers to the model that you want to train. the options are 
1) cnn_lstm_mm: hybrid (CNN+LSTM) multimodal model taking in input faces and poses with intermediate fusion between CNN and LSTM layers
2) cnn_lstm_mm_latefusion: hybrid (CNN+LSTM) multimodal model taking in input faces and poses with late fusion after the LSTM layers
3) cnn_lstm_image: hybrid (CNN+LSTM) model taking in input only faces
4) cnn_lstm_pose: hybrid (CNN+LSTM) model taking in input only poses

--data_dir', type=str, default='[your directory]/dataset_slots' : the directory where the dataset coming from the feature extraction of the Vernissage Corpus is saved.

--label_filename', type=str, default='3_classes.csv' : file with groundtruth labels referred to files in the dataset 
options are 
1) '3classes.csv' for models trained on three classes 'NAO', 'PLEFT', 'PRIGHT'
2) 'binary.csv' for models trained on binary classification, if the robot was 'INVOLVED' or 'NOT_INVOLVED'

--models_dir', type=str, default='[your directory]/models' : directory where models should be saved

--class_names', type=list, default=['NAO', 'PLEFT', 'PRIGHT'] : name of the classes
options are
1) 3classes ['NAO', 'PLEFT', 'PRIGHT']
2) binary ['INVOLVED', 'NOT_INVOLVED']

--solo_audio', type=bool, default=False : boolean variable to set the training of the model only on audio -  not used in the current project, leave to False

--all_slots_training', type=bool, default=True : boolean variable to set the training of the model on the entire dataset, following the 10-fold cross validation approach or only on a specific train- and test-set

--slot_test', type=int, default=0 : this variable needs to be defined only if all_slots_training is set to True. It refers to the slot you want to use as a test-set, leaving it out from the training phase. Slots are folders in data_dir

--seq_len', type=int, default=10 : length of the sequence of faces and poses you want to train your model on. For this project, do not change this parameters.

--n_seq', type=int, default=10 : number of sequences you want to train your model on: seq_len x n_seq = batch_size

--num_epochs', type=int, default=50 : number of epochs you want to train your model for

--hidden_dim', type=int, default=512 : hidden dimension of the LSTM

--layer_dim', type=int, default=1 : layer dimension of the LSTM

--learning_rate', type=float, default=0.001 : learning rate during optimization 

--patience', type=int, default=10 : number of epochs for early stopping

--step_size', type=int, default=40 : step size of the training, how many epochs to wait before applying a new learning rate adjustment

--optimizer1', type=str, default='SGD_opt')  : used for cnn_lstm_image
--optimizer2', type=str, default='Adam_opt') : used for all other models
    # options
    # SGD_opt 
    # RMSprop_opt 
    # Adam_opt
    
--dict_dir', type=str, default='' : directory where the performance of the training are saved

--training_complete', type=bool, default=False : boolean variable if you want to train your model on the entire dataset, leaving no data out. Set it True if you want to train models to be used on other data.

It takes in input: 
- Annotation files for each person of the Interaction (person1 and person2): "annotations_person{}.xml" 
- .txt files containing the timestamps of each raw data file (video and audio): "timestamp-naoaudio-begin.txt", 'timestamp-naoav-merged-begin.txt', 'naovideo-stats.txt'

It gives as output:
- .csv file with the annotation of the addressee, the time in sec., and the related timestamp for each raw data file: 'dataset_person{}.csv'


## Support
If you need any support, please write an email to carlo.mazzola@iit.it https://orcid.org/0000-0002-9282-9873
 
## Authors and acknowledgment

Author of the code:
Carlo Mazzola https://orcid.org/0000-0002-9282-9873

Acknoweledgement:
Thanks to Marta Romeo (second author of the paper) for her support in designing the architecture and in several steps of the development 

## License
Creative Commons Attribution 4.0 International (CC-BY-4.0)
https://joinup.ec.europa.eu/licence/creative-commons-attribution-40-international-cc-40

For attribution, cite the paper 
C. Mazzola, M. Romeo, F. Rea, A. Sciutti and A. Cangelosi, "To Whom are You Talking? A Deep Learning Model to Endow Social Robots with Addressee Estimation Skills," 2023 International Joint Conference on Neural Networks (IJCNN), Gold Coast, Australia, 2023, pp. 1-10, 
https://doi.org/10.1109/IJCNN54540.2023.10191452



