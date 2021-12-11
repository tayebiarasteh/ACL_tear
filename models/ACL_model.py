"""
Created on December 11, 2021.
ACL_model.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""

import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F


class ACL_net(nn.Module):
    def __init__(self, n_in_channels=1):
        super().__init__()
        first_feature_size = 8
        self.conv_block1 = Conv_block(n_in_channels, first_feature_size)
        self.conv_block2 = Conv_block(first_feature_size, first_feature_size * 2)
        self.squeeze_block1 = Squeeze_block(first_feature_size * 2, first_feature_size * 4)
        self.squeeze_block2 = Squeeze_block(first_feature_size * 4, first_feature_size * 8)
        self.squeeze_block3 = Final_Squeeze_block(first_feature_size * 8, first_feature_size * 8)
        self.attention_block1 = Attention_module(first_feature_size * 4, first_feature_size * 4)
        self.attention_block2 = Attention_module(first_feature_size * 8, first_feature_size * 8)
        self.attention_block2 = Output_block(first_feature_size * 8 + first_feature_size * 4,
                                             first_feature_size * 8 + first_feature_size * 4)

    def forward(self, input_tensor):
        conv_output1 = self.conv_block1(input_tensor)
        conv_output2 = self.conv_block2(conv_output1)
        squeeze_output1 = self.squeeze_block1(conv_output2)
        squeeze_output2 = self.squeeze_block2(squeeze_output1)
        squeeze_output_g = self.squeeze_block3(squeeze_output2)

        attention_output_zhat_1, attention_output_a_1 = self.attention_block1(squeeze_output1, squeeze_output_g)
        attention_output_zhat_2, attention_output_a_2 = self.attention_block2(squeeze_output2, squeeze_output_g)

        a_output = attention_output_a_1 + attention_output_a_2

        #Concatenation of the both paths
        attention_output_zhat_full = torch.cat((attention_output_zhat_1 , attention_output_zhat_2), 1)

        output_tensor = self.Output_block(attention_output_zhat_full)

        return output_tensor, a_output


class Conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.input_conv = nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=3)
        self.output_conv = nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)


    def forward(self, input_tensor):
        output_tensor = self.iterator(input_tensor, self.input_conv)
        output_tensor = F.relu(output_tensor)
        output_tensor = self.iterator(output_tensor, self.output_conv)
        output_tensor = F.relu(output_tensor)
        output_tensor = self.iterator(output_tensor, self.pool)
        return output_tensor


    def iterator(self, input_tensor, layer):
        temp = []
        for i in range(input_tensor.shape[2]):
            temp.append(layer(input_tensor[:, i]))
        input_tensor = torch.stack(temp, dim=2)
        return input_tensor



class Squeeze_module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1),
            nn.ReLU(inplace=True))
        self.conv2d = nn.Sequential(
            nn.Conv2d(int(out_ch/2), int(out_ch/2), kernel_size=3, padding=1))
        self.conv3d = nn.Sequential(
            nn.Conv3d(int(out_ch/2), int(out_ch/2), kernel_size=1))

    def forward(self, input_tensor):
        output_tensor = self.input_conv(input_tensor)

        above_path = self.iterator(output_tensor[:, output_tensor.shape[1]/2:], self.conv2d)
        below_path = self.conv3d(output_tensor[:, :output_tensor.shape[1]/2])

        #Concatenation of the both paths
        output_tensor = torch.cat((above_path , below_path), 1)
        output_tensor = F.relu(output_tensor)

        return output_tensor


    def iterator(self, input_tensor, layer):
        temp = []
        for i in range(input_tensor.shape[2]):
            temp.append(layer(input_tensor[:, i]))
        input_tensor = torch.stack(temp, dim=2)
        return input_tensor


class Attention_module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1),
            nn.ReLU(inplace=True))
        self.oneone_conv = nn.Sequential(
            nn.Conv3d(in_ch, 1, kernel_size=1),
            nn.ReLU(inplace=True))

    def forward(self, input_tensor_z, input_tensor_g):
        output_tensor_z = self.input_conv(input_tensor_z)
        output_tensor = output_tensor_z + input_tensor_g
        output_tensor = F.relu(output_tensor)
        output_tensor = self.oneone_conv(output_tensor)
        output_tensor_a = F.sigmoid(output_tensor)
        output_tensor_z_hat = input_tensor_z * output_tensor_a

        return output_tensor_z_hat, output_tensor_a


class Squeeze_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.input_squeeze = Squeeze_module(in_ch, out_ch)
        self.output_squeeze = Squeeze_module(out_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, input_tensor):
        output_tensor = self.input_squeeze(input_tensor)
        output_tensor = self.output_squeeze(output_tensor)
        output_tensor = self.iterator(output_tensor, self.pool)
        return output_tensor


    def iterator(self, input_tensor, layer):
        temp = []
        for i in range(input_tensor.shape[2]):
            temp.append(layer(input_tensor[:, i]))
        input_tensor = torch.stack(temp, dim=2)
        return input_tensor


class Final_Squeeze_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.input_squeeze = Final_Squeeze_module(in_ch, out_ch)
        self.output_squeeze = Final_Squeeze_module(out_ch, out_ch)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, input_tensor):
        output_tensor = self.input_squeeze(input_tensor)
        output_tensor = self.output_squeeze(output_tensor)
        output_tensor = self.pool(output_tensor)
        return output_tensor


class Final_Squeeze_module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1),
            nn.ReLU(inplace=True))
        self.conv3d_above = nn.Sequential(
            nn.Conv3d(int(out_ch/2), int(out_ch/2), kernel_size=3, padding=1))
        self.conv3d_below = nn.Sequential(
            nn.Conv3d(int(out_ch/2), int(out_ch/2), kernel_size=1))

    def forward(self, input_tensor):
        output_tensor = self.input_conv(input_tensor)

        above_path = self.conv3d_above(output_tensor[:, output_tensor.shape[1]/2:])
        below_path = self.conv3d_below(output_tensor[:, :output_tensor.shape[1]/2])

        #Concatenation of the both paths
        output_tensor = torch.cat((above_path , below_path), 1)
        output_tensor = F.relu(output_tensor)

        return output_tensor


class Output_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fully = nn.Linear(in_ch, out_ch)

    def forward(self, input_tensor):
        output_tensor = self.AdaptiveAvgPool3d(input_tensor)
        output_tensor = self.fully(output_tensor)
        return output_tensor


# up block 1
class Res_up_block_one(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch))
        self.up_sample = nn.Upsample(scale_factor=3)
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch))
        self.double_conv = double_conv(in_ch, out_ch)

    def forward(self, down_input, skip_input):
        down_input = self.up_sample(down_input)
        if skip_input.shape[-1] != down_input.shape[-1]:
            diff =  down_input.shape[-1] - skip_input.shape[-1]
            skip_input = F.pad(skip_input, (0, diff), "constant", 0)
        if skip_input.shape[-2] != down_input.shape[-2]:
            diff2 =  down_input.shape[-2] - skip_input.shape[-2]
            skip_input = F.pad(skip_input, (0, 0, 0, diff2), "constant", 0)

        input_tensor = torch.cat([down_input, skip_input], dim=1)
        identity = self.ch_avg(input_tensor)
        out = self.double_conv(input_tensor)
        output_tensor = out + identity
        return output_tensor

# up block 2,3
class Res_up_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch))
        self.up_sample = nn.Upsample(scale_factor=2)
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch))
        self.double_conv = double_conv(in_ch, out_ch)

    def forward(self, down_input, skip_input):
        down_input = self.up_sample(down_input)
        if skip_input.shape[-1] != down_input.shape[-1]:
            diff =  down_input.shape[-1] - skip_input.shape[-1]
            skip_input = F.pad(skip_input, (0, diff), "constant", 0)
        if skip_input.shape[-2] != down_input.shape[-2]:
            diff2 =  down_input.shape[-2] - skip_input.shape[-2]
            skip_input = F.pad(skip_input, (0, 0, 0, diff2), "constant", 0)

        input_tensor = torch.cat([down_input, skip_input], dim=1)
        identity = self.ch_avg(input_tensor)
        out = self.double_conv(input_tensor)
        output_tensor = out + identity
        return output_tensor




if __name__ == '__main__':
    asdf= 3