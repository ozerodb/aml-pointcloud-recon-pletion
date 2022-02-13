import torch
import torch.nn as nn
from . import encoder, decoder
import torch.nn.functional as F

class Generic_AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Generic_AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        BS, N, dim = x.size()
        assert dim == 3, "Fail: expecting 3 (x-y-z) as last tensor dimension!"

        # Refactoring batch
        x = x.permute(0, 2, 1)  # [BS, N, 3] => [BS, 3, N]

        # Encoding
        code = self.encoder(x)  # [BS, 3, N] => [BS, encoding_length]

        # Decoding
        decoded = self.decoder(code)  # [BS, 3, num_output_points]

        # Reshaping decoded output
        decoded = decoded.permute(0,2,1)  # [BS, 3, num_output_points] => [BS, num_output_points, 3]

        return decoded

class DownSampler(nn.Module):
    def __init__(self, input_points=1024, output_points=256):
        super(DownSampler, self).__init__()
        self.fc1 = nn.Linear(input_points, 512)
        self.bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, output_points)

    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        x = self.fc2(x)
        return x

def build_model(enc_type, encoding_length, dec_type, method, remove_point_num=None):
    # Encoder definition
    if enc_type == 'pointnet':
        enc = encoder.PointNet_encoder()
    elif enc_type == 'pointnetp1':
        enc = encoder.PointNetP1_encoder()
    elif enc_type == 'dgcnn' or enc_type == 'dgcnn_glob':
        enc = encoder.DGCNN_encoder()
    elif enc_type == 'dgcnn_loc':
        enc = encoder.DGCNN_encoder(k=6, pooling_type="avg")
    elif enc_type == 'dg':
        enc = encoder.DG_encoder()
    else:
        pass    # this shouldn't happen

    if encoding_length < 1024:
        enc = torch.nn.Sequential(
            enc,
            DownSampler(1024, encoding_length)
        )

    # Decoder definition
    if method == 'total':
        num_output_points=1024
    elif method == 'missing':
        num_output_points=remove_point_num

    if dec_type == 'fcs':
        dec = decoder.FCsmall_decoder(encoding_length=encoding_length, num_output_points=num_output_points)
    if dec_type == 'fcm':
        dec = decoder.FCmedium_decoder(encoding_length=encoding_length, num_output_points=num_output_points)
    if dec_type == 'fcl':
        dec = decoder.FClarge_decoder(encoding_length=encoding_length, num_output_points=num_output_points)
    elif dec_type == 'ppd':
        dec = decoder.PPD_decoder(encoding_length=encoding_length, num_output_points=num_output_points)
    else:
        pass    # this shouldn't happen

    # Putting everything together
    model = Generic_AutoEncoder(enc, dec)
    return model