import torch
import torch.nn as nn
import torch.nn.functional as F

class FCsmall_decoder(nn.Module):
    """
    A simple fully-connected decoder. Small version (suggested for code sizes <= 256).

    Given a feature vector, it inferes a set of xyz coordinates, i.e., a point cloud.
    """
    def __init__(self, encoding_length = 128, num_output_points = 1024):
        """
        Parameters
        ----------
        encoding_length: int, optional
            the lenght of the feature vector (default is 128)
        num_output_points: int, optional
            the desired number of output points (default is 1024)
        """
        super(FCsmall_decoder, self).__init__()
        self.num_output_points = num_output_points
        self.fc1 = nn.Linear(encoding_length, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.2)
        self.fc4 = nn.Linear(1024, self.num_output_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, self.num_output_points)
        return x

class FCmedium_decoder(nn.Module):
    """
    A simple fully-connected decoder. Medium version (suggested for code sizes <= 512).

    Given a feature vector, it inferes a set of xyz coordinates, i.e., a point cloud.
    """
    def __init__(self, encoding_length = 256, num_output_points = 1024):
        """
        Parameters
        ----------
        encoding_length: int, optional
            the lenght of the feature vector (default is 256)
        num_output_points: int, optional
            the desired number of output points (default is 1024)
        """
        super(FCmedium_decoder, self).__init__()
        self.num_output_points = num_output_points
        self.fc1 = nn.Linear(encoding_length, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.2)
        self.fc4 = nn.Linear(1024, self.num_output_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, self.num_output_points)
        return x

class FClarge_decoder(nn.Module):
    """
    A simple fully-connected decoder. Large version (suggested for code sizes <= 1024).

    Given a feature vector, it inferes a set of xyz coordinates, i.e., a point cloud.
    """
    def __init__(self, encoding_length = 512, num_output_points = 1024):
        """
        Parameters
        ----------
        encoding_length: int, optional
            the lenght of the feature vector (default is 512)
        num_output_points: int, optional
            the desired number of output points (default is 1024)
        """
        super(FClarge_decoder, self).__init__()
        self.num_output_points = num_output_points
        self.fc1 = nn.Linear(encoding_length, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.2)
        self.fc4 = nn.Linear(1024, self.num_output_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, self.num_output_points)
        return x

class PPD_decoder(nn.Module):
    """
    Adaptation of the Point Pyramid decoder used in PF-Net.

    Given a feature vector, it inferes a set of xyz coordinates, i.e., a point cloud.
    """
    def  __init__(self, encoding_length = 512, num_output_points = 1024):
        """
        Parameters
        ----------
        encoding_length: int, optional
            the lenght of the feature vector (default is 512)
        num_output_points: int, optional
            the desired number of output points (default is 1024)
        """
        super(PPD_decoder,self).__init__()
        self.num_output_points = num_output_points

        self.fc1 = nn.Linear(encoding_length, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        self.fc3_1 = nn.Linear(128, 64*3)
        self.fc2_1 = nn.Linear(256, 64*128)
        self.fc1_1 = nn.Linear(512, 128*512)

        self.conv1_1 = torch.nn.Conv1d(512, 512, 1)
        self.conv1_2 = torch.nn.Conv1d(512, 256, 1)
        self.conv1_3 = torch.nn.Conv1d(256, int((self.num_output_points*3)/128), 1)
        self.conv2_1 = torch.nn.Conv1d(128, 6, 1)

    def forward(self, x):
        x_1 = F.relu(self.fc1(x)) # 512
        x_2 = F.relu(self.fc2(x_1)) # 256
        x_3 = F.relu(self.fc3(x_2))  # 128

        pc1_feat = self.fc3_1(x_3)
        pc1_xyz = pc1_feat.reshape(-1, 64, 3)

        pc2_feat = F.relu(self.fc2_1(x_2))
        pc2_feat = pc2_feat.reshape(-1, 128, 64)
        pc2_xyz =self.conv2_1(pc2_feat)

        pc3_feat = F.relu(self.fc1_1(x_1))
        pc3_feat = pc3_feat.reshape(-1, 512, 128)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        pc3_xyz = self.conv1_3(pc3_feat)

        pc1_xyz_expand = torch.unsqueeze(pc1_xyz, 2)
        pc2_xyz = pc2_xyz.transpose(1, 2)
        pc2_xyz = pc2_xyz.reshape(-1, 64, 2, 3)
        pc2_xyz = pc1_xyz_expand + pc2_xyz
        pc2_xyz = pc2_xyz.reshape(-1, 128, 3)

        pc2_xyz_expand = torch.unsqueeze(pc2_xyz, 2)
        pc3_xyz = pc3_xyz.transpose(1, 2)
        pc3_xyz = pc3_xyz.reshape(-1, 128, int(self.num_output_points/128), 3)
        pc3_xyz = pc2_xyz_expand + pc3_xyz
        pc3_xyz = pc3_xyz.reshape(-1, self.num_output_points, 3)

        decoded = pc3_xyz.permute(0, 2, 1)  # [BS, num_output_points, 3] => [BS, 3, num_output_points]
        return decoded