import torch
import torch.nn as nn
from torch.optim import lr_scheduler


torch.manual_seed(0)


# CNN model for images
class Conv_NN_Image(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self):
        super(Conv_NN_Image, self).__init__()

        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=7)  # 1
        self.relu1 = nn.LeakyReLU()
        self.conv_layer2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=5)  # 2
        self.relu2 = nn.LeakyReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 3

        self.conv_layer3 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=5)  # 4
        self.relu3 = nn.LeakyReLU()
        self.conv_layer4 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3)  # 5
        self.relu4 = nn.LeakyReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 6

        self.fc1 = nn.Linear(18496, 4624)  # 7 (9248, 1156) (18496, 4624)
        self.relu5 = nn.LeakyReLU()  # 8
        self.fc2 = nn.Linear(4624, 578)  # 9 (1156, 578)

    # Progresses data across layers
    def forward(self, x):
        #print("0face", x.shape)
        out = self.conv_layer1(x)
        #out = self.relu1(out)
        #print("conv1", out.shape)
        out = self.conv_layer2(out)
        #print("conv2", out.shape)
        out = self.relu2(out)
        #print("2", out.shape)
        out = self.max_pool1(out)
        #print("max1", out.shape)
        #print("3", out.shape)

        out = self.conv_layer3(out)
        #print("conv3", out.shape)
        #out = self.relu3(out)
        #print("4", out.shape)
        out = self.conv_layer4(out)
        #print("conv4", out.shape)
        out = self.relu4(out)
        #print("5", out.shape)
        out = self.max_pool2(out)
        #print("max2", out.shape)
        #print("6", out.shape)
        out = out.reshape(out.size(0), -1)
        #print("resize", out.shape)
        #print("7", out.shape)
        out = self.fc1(out)
        #print("fc1", out.shape)
        #print("8", out.shape)
        out = self.relu5(out)
        #print("9", out.shape)
        out = self.fc2(out)
        #print("fc2", out.shape)
        #print("10_face", out.shape)

        return out


# CNN model for poses
class Conv_NN_Pose(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self):
        super(Conv_NN_Pose, self).__init__()

        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1))
        self.relu1 = nn.LeakyReLU()
        self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 1))
        self.relu2 = nn.LeakyReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv_layer3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1))
        self.relu3 = nn.LeakyReLU()
        self.conv_layer4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1))
        self.relu4 = nn.LeakyReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(32, 24)
        self.relu5 = nn.LeakyReLU()
        self.fc2 = nn.Linear(24, 20)

    # Progresses data across layers
    def forward(self, x):
        #print("pose ", x.shape)
        out = self.conv_layer1(x)
        #out = self.relu1(out)
        #print("conv1", out.shape)
        out = self.conv_layer2(out)
        out = self.relu2(out)
        #print("conv2", out.shape)
        out = self.max_pool1(out)
        #print("max1", out.shape)

        out = self.conv_layer3(out)
        #out = self.relu3(out)
        #print("conv3", out.shape)
        out = self.conv_layer4(out)
        out = self.relu4(out)
        #print("conv4", out.shape)
        out = self.max_pool2(out)
        #print("max2", out.shape)
        out = out.reshape(out.size(0), -1)
        #print("reshape", out.shape)
        out = self.fc1(out)
        #print("fc1", out.shape)
        out = self.relu5(out)
        #print("9", out.shape)
        out = self.fc2(out)
        #print("fc2", out.shape)

        return out


# Creating a LSTM class
class LSTM_NN_Vision(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, hidden_dim, seq_len, layer_dim, n_seq, num_classes, device):
        super(LSTM_NN_Vision, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.seq_len = seq_len
        self.n_seq = n_seq
        self.device = device

        self.lstm = nn.LSTM(input_size=1158, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)  # 10
        #print(self.hidden_dim)
        # Readout layer
        self.fc3 = nn.Linear(self.hidden_dim, 128)  # 11
        self.relu1 = nn.LeakyReLU()
        self.fc4 = nn.Linear(128, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        #self.softmax = nn.Softmax(dim=1)

    # Progresses data across layers
    def forward(self, x):
        current_batch = x.view(-1, self.seq_len, x.size(1)).size(0)

        self.hidden = (torch.zeros(1 * self.layer_dim, current_batch, self.hidden_dim).to(self.device),
                       torch.zeros(1 * self.layer_dim, current_batch, self.hidden_dim).to(self.device))

        #print(x.view(current_batch, self.seq_len, x.size(1)).shape)
        lstm_out, self.hidden = self.lstm(x.view(current_batch, self.seq_len, x.size(1)), self.hidden)
        #print("lstm ", lstm_out[:, -1, :].shape)

        out = self.fc3(lstm_out[:, -1, :])
        #print("fc3", out.shape)
        out = self.relu1(out)
        #print("13", out.shape)
        out = self.fc4(out)
        #print("fc4", out.shape)
        y_pred = self.softmax(out)
        #print("softmax", y_pred.shape)

        return y_pred, current_batch


# Creating a LSTM class
class LSTM_NN_Image(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, hidden_dim, seq_len, layer_dim, n_seq, num_classes, device):
        super(LSTM_NN_Image, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.seq_len = seq_len
        self.n_seq = n_seq
        self.device = device

        self.lstm = nn.LSTM(input_size=578, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)  # 10

        # Readout layer
        self.fc3 = nn.Linear(self.hidden_dim, 128)  # 11
        self.relu1 = nn.LeakyReLU()
        self.fc4 = nn.Linear(128, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    # Progresses data across layers
    def forward(self, x):
        current_batch = x.view(-1, self.seq_len, x.size(1)).size(0)
        print(current_batch)

        self.hidden = (torch.zeros(1 * self.layer_dim, current_batch, self.hidden_dim).to(self.device),
                       torch.zeros(1 * self.layer_dim, current_batch, self.hidden_dim).to(self.device))
        print(x.view(current_batch, self.seq_len, x.size(1)).shape)
        lstm_out, self.hidden = self.lstm(x.view(current_batch, self.seq_len, x.size(1)), self.hidden)
        print("lstm", lstm_out[:, -1, :].shape)
        out = self.fc3(lstm_out[:, -1, :])
        print("fc3", out.shape)
        out = self.relu1(out)

        out = self.fc4(out)
        print("fc4", out.shape)
        y_pred = self.softmax(out)
        # print("14", y_pred.shape)

        return y_pred, current_batch


# Creating a LSTM class
class LSTM_NN_Pose(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, hidden_dim, seq_len, layer_dim, n_seq, num_classes, device):
        super(LSTM_NN_Pose, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.seq_len = seq_len
        self.n_seq = n_seq
        self.device = device

        self.lstm = nn.LSTM(input_size=20, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)  # 10

        # Readout layer
        self.fc3 = nn.Linear(self.hidden_dim, 128)  # 11  #128
        self.relu1 = nn.LeakyReLU()
        self.fc4 = nn.Linear(128, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    # Progresses data across layers
    def forward(self, x):
        current_batch = x.view(-1, self.seq_len, x.size(1)).size(0)
        print(current_batch)

        self.hidden = (torch.zeros(1 * self.layer_dim, current_batch, self.hidden_dim).to(self.device),
                       torch.zeros(1 * self.layer_dim, current_batch, self.hidden_dim).to(self.device))
        print(x.view(current_batch, self.seq_len, x.size(1)).shape)
        lstm_out, self.hidden = self.lstm(x.view(current_batch, self.seq_len, x.size(1)), self.hidden)
        print("lstm", lstm_out[:, -1, :].shape)
        out = self.fc3(lstm_out[:, -1, :])
        print("fc3", out.shape)
        out = self.relu1(out)

        out = self.fc4(out)
        print("fc4", out.shape)
        y_pred = self.softmax(out)
        # print("14", y_pred.shape)

        return y_pred, current_batch


# Creating a LSTM class
class LSTM_NN_Image_LateFusion(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, hidden_dim, seq_len, layer_dim, n_seq, num_classes, device):
        super(LSTM_NN_Image_LateFusion, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.seq_len = seq_len
        self.n_seq = n_seq
        self.device = device
        self.hidden_dim = 512

        self.lstm = nn.LSTM(input_size=578, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)  # 10

        # Readout layer
        self.fc3 = nn.Linear(self.hidden_dim, 128)  # 11
        #self.relu1 = nn.LeakyReLU()

    # Progresses data across layers
    def forward(self, x):
        current_batch = x.view(-1, self.seq_len, x.size(1)).size(0)
        print(current_batch)

        self.hidden = (torch.zeros(1 * self.layer_dim, current_batch, self.hidden_dim).to(self.device),
                       torch.zeros(1 * self.layer_dim, current_batch, self.hidden_dim).to(self.device))

        lstm_out, self.hidden = self.lstm(x.view(current_batch, self.seq_len, x.size(1)), self.hidden)
        print("lstm", lstm_out[:, -1, :].shape)
        out = self.fc3(lstm_out[:, -1, :])
        print("fc3", out.shape)
        #out = self.relu1(out)

        return out, current_batch


# Creating a LSTM class
class LSTM_NN_Pose_LateFusion(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, hidden_dim, seq_len, layer_dim, n_seq, num_classes, device):
        super(LSTM_NN_Pose_LateFusion, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.seq_len = seq_len
        self.n_seq = n_seq
        self.device = device
        self.hidden_dim = 256

        self.lstm = nn.LSTM(input_size=20, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)  # 10

        # Readout layer
        self.fc3 = nn.Linear(self.hidden_dim, 128)  # 11
        #self.relu1 = nn.LeakyReLU()

    # Progresses data across layers
    def forward(self, x):
        current_batch = x.view(-1, self.seq_len, x.size(1)).size(0)
        # print(current_batch)

        self.hidden = (torch.zeros(1 * self.layer_dim, current_batch, self.hidden_dim).to(self.device),
                       torch.zeros(1 * self.layer_dim, current_batch, self.hidden_dim).to(self.device))

        lstm_out, self.hidden = self.lstm(x.view(current_batch, self.seq_len, x.size(1)), self.hidden)
        print("lstm", lstm_out[:, -1, :].shape)

        out = self.fc3(lstm_out[:, -1, :])
        print("fc3", out.shape)
        # print("12", out.shape)
        #out = self.relu1(out)

        return out, current_batch


class FC_Late_Fusion(nn.Module):
    def __init__(self, num_classes):
        super(FC_Late_Fusion, self).__init__()

        self.fc1 = nn.Linear(256, 128)  # 11
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    #Progresses data across layers
    def forward(self, x):
        print(x.shape)
        out = self.fc1(x)
        print('fc1', out.shape)
        out = self.relu1(out)
        #print('relu1', out.shape)
        out = self.fc2(out)
        print('fc2', out.shape)
        y_pred = self.softmax(out)
        return y_pred

# CNN model for images
class Conv_NN_Audio(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, seq_len, num_classes):
        super(Conv_NN_Audio, self).__init__()
        self.seq_len = seq_len
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=9)  # 1
        self.conv_layer2 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=7)  # 2
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 3

        self.conv_layer3 = nn.Conv2d(in_channels=5, out_channels=7, kernel_size=7)  # 4
        self.conv_layer4 = nn.Conv2d(in_channels=7, out_channels=7, kernel_size=5)  # 5
        self.max_pool2 = nn.MaxPool2d(kernel_size=5, stride=5)  # 6

        self.fc1 = nn.Linear(1008, 128)  # 7
        self.relu1 = nn.ReLU()  # 8
        self.fc2 = nn.Linear(128, num_classes)  # 9

    # Progresses data across layers
    def forward(self, x):
        # print(self.seq_len, x.size(1))
        current_batch = x.shape[0]
        # print('current_batch', current_batch)
        # print("0", x.shape)
        out = self.conv_layer1(x)
        # print("1", out.shape)
        out = self.conv_layer2(out)
        # print("2", out.shape)
        out = self.max_pool1(out)
        # print("3", out.shape)

        out = self.conv_layer3(out)
        # print("4", out.shape)
        out = self.conv_layer4(out)
        # print("5", out.shape)
        out = self.max_pool2(out)
        # print("6", out.shape)
        out = out.reshape(out.size(0), -1)
        # print("7", out.shape)
        out = self.fc1(out)
        # print("8", out.shape)
        out = self.relu1(out)
        # print("9", out.shape)
        output = self.fc2(out)
        # print("10", output.shape)

        return output, current_batch


class LSTM_NN_Vision_with_other(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, hidden_dim, seq_len, layer_dim, n_seq, num_classes, device):
        super(LSTM_NN_Vision_with_other, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.seq_len = seq_len
        self.n_seq = n_seq
        self.device = device

        self.lstm = nn.LSTM(input_size=1636, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)  # 10

        # Readout layer
        self.fc3 = nn.Linear(self.hidden_dim, 256)  # 11
        self.relu1 = nn.ReLU()
        self.fc4 = nn.Linear(256, num_classes)

    # Progresses data across layers
    def forward(self, x):
        current_batch = x.view(-1, self.seq_len, x.size(1)).size(0)
        # print(current_batch)

        self.hidden = (torch.zeros(1 * self.layer_dim, current_batch, self.hidden_dim).to(self.device),
                       torch.zeros(1 * self.layer_dim, current_batch, self.hidden_dim).to(self.device))

        lstm_out, self.hidden = self.lstm(x.view(current_batch, self.seq_len, x.size(1)), self.hidden)

        out = self.fc3(lstm_out[:, -1, :])
        # print("12", out.shape)
        out = self.relu1(out)
        # print("13", out.shape)
        y_pred = self.fc4(out)
        # print("14", y_pred.shape)

        return y_pred, current_batch


class LSTM_NN_Vision_Audio(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, hidden_dim, seq_len, layer_dim, n_seq, num_classes, device):
        super(LSTM_NN_Vision_Audio, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.seq_len = seq_len
        self.n_seq = n_seq
        self.device = device

        self.lstm = nn.LSTM(input_size=1736, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)  # 10

        # Readout layer
        self.fc3 = nn.Linear(self.hidden_dim, 512)  # 11
        self.relu1 = nn.ReLU()
        self.fc4 = nn.Linear(512, num_classes)

    # Progresses data across layers
    def forward(self, x):
        current_batch = x.view(-1, self.seq_len, x.size(1)).size(0)

        self.hidden = (torch.zeros(1 * self.layer_dim, current_batch, self.hidden_dim).to(self.device),
                       torch.zeros(1 * self.layer_dim, current_batch, self.hidden_dim).to(self.device))

        lstm_out, self.hidden = self.lstm(x.view(current_batch, self.seq_len, x.size(1)), self.hidden)

        out = self.fc3(lstm_out[:, -1, :])
        # print("12", out.shape)
        out = self.relu1(out)
        # print("13", out.shape)
        y_pred = self.fc4(out)
        # print("14", y_pred.shape)

        return y_pred, current_batch


# Creating a LSTM class
class LSTM_NN_Audio(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, hidden_dim, seq_len, layer_dim, n_seq, num_classes, device):
        super(LSTM_NN_Audio, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.seq_len = seq_len
        self.n_seq = n_seq
        self.device = device

        self.lstm = nn.LSTM(input_size=578, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)  # 10

        # Readout layer
        self.fc3 = nn.Linear(self.hidden_dim, 128)  # 11
        self.relu1 = nn.ReLU()
        self.fc4 = nn.Linear(128, num_classes)

    # Progresses data across layers
    def forward(self, x):
        current_batch = x.view(-1, self.seq_len, x.size(1)).size(0)
        # print(current_batch)

        self.hidden = (torch.zeros(1 * self.layer_dim, current_batch, self.hidden_dim).to(self.device),
                       torch.zeros(1 * self.layer_dim, current_batch, self.hidden_dim).to(self.device))

        lstm_out, self.hidden = self.lstm(x.view(current_batch, self.seq_len, x.size(1)), self.hidden)

        out = self.fc3(lstm_out[:, -1, :])
        # print("12", out.shape)
        out = self.relu1(out)
        # print("13", out.shape)
        y_pred = self.fc4(out)
        # print("14", y_pred.shape)

        return y_pred, current_batch



def build_models(d):
    device = d['device']
    model_cnn_image = Conv_NN_Image().to(device)
    model_cnn_pose = Conv_NN_Pose().to(device)
    model_cnn_audio = Conv_NN_Audio(d['seq_len'], d['num_classes']).to(device)
    model_lstm_vision = LSTM_NN_Vision(d['hidden_dim'], d['seq_len'], d['layer_dim'], d['n_seq'], d['num_classes'],
                                       device).to(device)

    model_lstm_image_latefusion = LSTM_NN_Image_LateFusion(d['hidden_dim'], d['seq_len'], d['layer_dim'], d['n_seq'], d['num_classes'],
                                     device).to(device)
    model_lstm_pose_latefusion = LSTM_NN_Pose_LateFusion(d['hidden_dim'], d['seq_len'], d['layer_dim'], d['n_seq'], d['num_classes'],
                                   device).to(device)
    model_fc_latefusion = FC_Late_Fusion(d['num_classes']).to(device)

    model_lstm_vision_with_other = LSTM_NN_Vision_with_other(d['hidden_dim'], d['seq_len'], d['layer_dim'], d['n_seq'],
                                                             d['num_classes'], device).to(device)
    model_lstm_vision_audio = LSTM_NN_Vision_Audio(d['hidden_dim'], d['seq_len'], d['layer_dim'], d['n_seq'],
                                                             d['num_classes'], device).to(device)
    model_lstm_audio = LSTM_NN_Audio(d['hidden_dim'], d['seq_len'], d['layer_dim'], d['n_seq'], d['num_classes'],
                                     device).to(device)
    model_lstm_image = LSTM_NN_Image(d['hidden_dim'], d['seq_len'], d['layer_dim'], d['n_seq'], d['num_classes'],
                                     device).to(device)
    model_lstm_pose = LSTM_NN_Pose(d['hidden_dim'], d['seq_len'], d['layer_dim'], d['n_seq'], d['num_classes'],
                                   device).to(device)

    cnn_image_d = {
        'name': 'cnn_image',
        'model': model_cnn_image,
        'SGD_opt': torch.optim.SGD(model_cnn_image.parameters(), lr=d['learning_rate'],
                                   weight_decay=0.005, momentum=0.9),# weight_decay = 0.001
        'Adam_opt': torch.optim.Adam(model_cnn_image.parameters(), lr=d['learning_rate'],
                                     weight_decay=0.000),  # weight_decay = 0.001
        'RMSprop_opt': torch.optim.RMSprop(model_cnn_image.parameters(), lr=d['learning_rate'],
                                           alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    }
    cnn_pose_d = {
        'name': 'cnn_pose',
        'model': model_cnn_pose,
        'SGD_opt': torch.optim.SGD(model_cnn_pose.parameters(), lr=d['learning_rate'],
                                   weight_decay=0.005, momentum=0.9),
        'Adam_opt': torch.optim.Adam(model_cnn_pose.parameters(), lr=d['learning_rate'],
                                     weight_decay=0.000),  # weight_decay = 0.001
        'RMSprop_opt': torch.optim.RMSprop(model_cnn_pose.parameters(), lr=d['learning_rate'],
                                           alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    }
    cnn_audio_d = {
        'name': 'cnn_audio',
        'model': model_cnn_audio,
        'SGD_opt': torch.optim.SGD(model_cnn_audio.parameters(), lr=d['learning_rate'],
                                   weight_decay=0.005, momentum=0.9),
        'Adam_opt': torch.optim.Adam(model_cnn_audio.parameters(), lr=d['learning_rate'],
                                     weight_decay=0.000),  # weight_decay = 0.001
        'RMSprop_opt': torch.optim.RMSprop(model_cnn_audio.parameters(), lr=d['learning_rate'],
                                           alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    }
    lstm_vision_d = {
        'name': 'lstm_vision',
        'model': model_lstm_vision,
        'SGD_opt': torch.optim.SGD(model_lstm_vision.parameters(), lr=d['learning_rate'],
                                   weight_decay=0.005, momentum=0.9),
        'Adam_opt': torch.optim.Adam(model_lstm_vision.parameters(), lr=d['learning_rate'],
                                     weight_decay=0.000),  # weight_decay = 0.001
        'RMSprop_opt': torch.optim.RMSprop(model_lstm_vision.parameters(), lr=d['learning_rate'],
                                           alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    }
    lstm_image_d_latefusion = {
        'name': 'lstm_image_latefusion',
        'model': model_lstm_image_latefusion,
        'SGD_opt': torch.optim.SGD(model_lstm_image.parameters(), lr=d['learning_rate'],
                                   weight_decay=0.005, momentum=0.9),
        'Adam_opt': torch.optim.Adam(model_lstm_image.parameters(), lr=d['learning_rate'],
                                     weight_decay=0.000),  # weight_decay = 0.001
        'RMSprop_opt': torch.optim.RMSprop(model_lstm_image.parameters(), lr=d['learning_rate'],
                                           alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    }
    lstm_pose_d_latefusion = {
        'name': 'lstm_pose_latefusion',
        'model': model_lstm_pose_latefusion,
        'SGD_opt': torch.optim.SGD(model_lstm_pose.parameters(), lr=d['learning_rate'],
                                   weight_decay=0.005, momentum=0.9),
        'Adam_opt': torch.optim.Adam(model_lstm_pose.parameters(), lr=d['learning_rate'],
                                     weight_decay=0.000),  # weight_decay = 0.001
        'RMSprop_opt': torch.optim.RMSprop(model_lstm_pose.parameters(), lr=d['learning_rate'],
                                           alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    }
    fc_latefusion = {
        'name': 'fc_latefusion',
        'model': model_fc_latefusion,
        'SGD_opt': torch.optim.SGD(model_lstm_pose.parameters(), lr=d['learning_rate'],
                                   weight_decay=0.005, momentum=0.9),
        'Adam_opt': torch.optim.Adam(model_lstm_pose.parameters(), lr=d['learning_rate'],
                                     weight_decay=0.000),  # weight_decay = 0.001
        'RMSprop_opt': torch.optim.RMSprop(model_lstm_pose.parameters(), lr=d['learning_rate'],
                                           alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    }
    lstm_vision_with_other_d = {
        'name': 'lstm_vision_with_other',
        'model': model_lstm_vision_with_other,
        'SGD_opt': torch.optim.SGD(model_lstm_vision_with_other.parameters(), lr=d['learning_rate'],
                                   weight_decay=0.005, momentum=0.9),
        'Adam_opt': torch.optim.Adam(model_lstm_vision_with_other.parameters(), lr=d['learning_rate'],
                                     weight_decay=0.000),  # weight_decay = 0.001
        'RMSprop_opt': torch.optim.RMSprop(model_lstm_vision_with_other.parameters(), lr=d['learning_rate'],
                                           alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    }
    lstm_vision_audio_d =  {
        'name': 'lstm_vision_audio',
        'model': model_lstm_vision_audio,
        'SGD_opt': torch.optim.SGD(model_lstm_vision_audio.parameters(), lr=d['learning_rate'],
                                   weight_decay=0.005, momentum=0.9),
        'Adam_opt': torch.optim.Adam(model_lstm_vision_audio.parameters(), lr=d['learning_rate'],
                                     weight_decay=0.000),  # weight_decay = 0.001
        'RMSprop_opt': torch.optim.RMSprop(model_lstm_vision_audio.parameters(), lr=d['learning_rate'],
                                           alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    }
    lstm_audio_d = {
        'name': 'lstm_audio',
        'model': model_lstm_audio,
        'SGD_opt': torch.optim.SGD(model_lstm_audio.parameters(), lr=d['learning_rate'],
                                   weight_decay=0.005, momentum=0.9),
        'Adam_opt': torch.optim.Adam(model_lstm_audio.parameters(), lr=d['learning_rate'],
                                     weight_decay=0.000),  # weight_decay = 0.001
        'RMSprop_opt': torch.optim.RMSprop(model_lstm_audio.parameters(), lr=d['learning_rate'],
                                           alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    }
    lstm_image_d = {
        'name': 'lstm_image',
        'model': model_lstm_image,
        'SGD_opt': torch.optim.SGD(model_lstm_image.parameters(), lr=d['learning_rate'],
                                   weight_decay=0.005, momentum=0.9),
        'Adam_opt': torch.optim.Adam(model_lstm_image.parameters(), lr=d['learning_rate'],
                                     weight_decay=0.000),  # weight_decay = 0.001
        'RMSprop_opt': torch.optim.RMSprop(model_lstm_image.parameters(), lr=d['learning_rate'],
                                           alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    }
    lstm_pose_d = {
        'name': 'lstm_pose',
        'model': model_lstm_pose,
        'SGD_opt': torch.optim.SGD(model_lstm_pose.parameters(), lr=d['learning_rate'],
                                   weight_decay=0.005, momentum=0.9),
        'Adam_opt': torch.optim.Adam(model_lstm_pose.parameters(), lr=d['learning_rate'],
                                     weight_decay=0.000),  # weight_decay = 0.001
        'RMSprop_opt': torch.optim.RMSprop(model_lstm_pose.parameters(), lr=d['learning_rate'],
                                           alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    }

    all_models = [cnn_image_d, cnn_pose_d, cnn_audio_d, lstm_vision_d, lstm_vision_with_other_d, lstm_vision_audio_d, lstm_audio_d,
                  lstm_image_d, lstm_pose_d, lstm_image_d_latefusion, lstm_pose_d_latefusion, fc_latefusion]

    for model in all_models:
        model['scheduler1'] = lr_scheduler.StepLR(model[d['optimizer1']], step_size=d['step_size'], gamma=0.1)
        model['scheduler2'] = lr_scheduler.StepLR(model[d['optimizer2']], step_size=d['step_size'], gamma=0.1)
        model['criterion'] = nn.NLLLoss()
        #model['criterion'] = nn.CrossEntropyLoss()

        for i in range(len(list(model['model'].parameters()))):
            print(list(model['model'].parameters())[i].size())

    return cnn_image_d, cnn_pose_d, cnn_audio_d, lstm_vision_d, lstm_vision_with_other_d, lstm_vision_audio_d, lstm_audio_d, lstm_image_d, lstm_pose_d, lstm_image_d_latefusion, lstm_pose_d_latefusion, fc_latefusion
