import torch
from torch import nn
import numpy as np
import pandas as pd
import os
import pydicom as dicom
import math
import time

# Positional encoding class
# May want to scrap this for a learnable positional encoding model as opposed to sinusoidal


class PositionalEncoding(nn.Module):
    def __init__(self, data, dropout=0.1, n=10000):
        super(PositionalEncoding, self).__init__()
        self.embedded_dim, self.position = data.shape
        self.dropout = nn.Dropout(p=dropout)

        self.embedded_dim += 1  # adding one to embedded dim to take into account token prepend

        self.learned_embedding_vec = nn.Parameter(
            torch.zeros(1, self.position))

        self.positional_matrix = torch.zeros(self.embedded_dim, self.position)

        for pos in range(self.position):
            for i in range(int(self.embedded_dim/2)):
                denom = pow(n, 2*i/self.embedded_dim)
                self.positional_matrix[2*i, pos] = np.sin(pos/denom)
                self.positional_matrix[2*i+1, pos] = np.cos(pos/denom)

    def forward(self, data):
        data = torch.vstack((self.learned_embedding_vec, data))
        summer_matrix = data + self.positional_matrix
        summer_matrix = self.dropout(summer_matrix)

        return self.summer_matrix


# Apply conv layer to a (500, 400) subset of each scan
# TODO max pool is necessary
class ConvLayer(nn.Module):
    def __init__(self, num_patch: int = 49):
        super(ConvLayer, self).__init__()
        self.num_patch = num_patch
        n = num_patch
        self.conv2d_1 = nn.Conv2d(
            in_channels=n*1, out_channels=n * 8, kernel_size=13, stride=1, groups=n)
        self.pooling2d_1 = nn.MaxPool2d(2)
        self.conv2d_2 = nn.Conv2d(
            in_channels=n*8, out_channels=n*16, kernel_size=11, stride=1, groups=n)
        self.pooling2d_2 = nn.MaxPool2d(2)
        self.conv2d_3 = nn.Conv2d(
            in_channels=n*16, out_channels=n*32, kernel_size=9, stride=1, groups=n)
        self.conv2d_4 = nn.Conv2d(
            in_channels=n*32, out_channels=n*32, kernel_size=7, stride=1, groups=n)
        self.pooling2d_3 = nn.MaxPool2d(2)
        self.conv2d_5 = nn.Conv2d(
            in_channels=n*32, out_channels=n*64, kernel_size=5, stride=1, groups=n)
        self.dnn = nn.Linear(105280, 256)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, tensor):
        x = self.conv2d_1(tensor)
        x = self.relu(x)
        x = self.pooling2d_1(x)

        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.pooling2d_2(x)

        x = self.conv2d_3(x)
        x = self.relu(x)
        x = self.conv2d_4(x)
        x = self.relu(x)

        x = self.pooling2d_3(x)

        x = self.conv2d_5(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = torch.reshape(x, (self.num_patch, 105280))
        x = self.dnn(x)

        return x


# Full Embedding class that takes in data name for individual image and outputs positional embedding where each column
# vector represents a positionally-embedded patch except for the very first column vector, which is a learnt
# classification token

class EmbeddingBlock(nn.Module):
    # Data in this sense is the image that has not been translated into an array
    # Want to set x_con to 3500
    def __init__(self, x_amount=7, y_amount=7, x_con=3500, y_con=2800):
        super(EmbeddingBlock, self).__init__()

        assert (x_con % x_amount == 0)
        assert (y_con % y_amount == 0)
        self.x_amount = x_amount
        self.y_amount = y_amount
        self.x_con = x_con
        self.y_con = y_con

        self.amount_of_patches = int(x_amount * y_amount)
        self.x_ran = int(x_con / x_amount)
        self.y_ran = int(y_con / y_amount)
        self.patches_matrix = torch.zeros(
            self.amount_of_patches, self.x_ran, self.y_ran)

        self.cc_conv = ConvLayer()
        self.mlo_conv = ConvLayer()

    def forward(self, data):
        # recheck for proper class variables(change self. to strictly local variable)
        info = data[:, :self.x_con, :self.y_con]

        batch_size = info.shape[0]

        batched_patches = info.unfold(
            1, self.x_ran, self.x_ran).unfold(2, self.y_ran, self.y_ran)
        batched_patches = torch.reshape(batched_patches,
                                        (batch_size, self.amount_of_patches, self.x_ran, self.y_ran))

        LCC = batched_patches[0]
        LMLO = batched_patches[1]
        RCC = batched_patches[2]
        RMLO = batched_patches[3]

        LCC = self.cc_conv.forward(LCC)
        RCC = self.cc_conv.forward(RCC)
        LMLO = self.mlo_conv.forward(LMLO)
        RMLO = self.mlo_conv.forward(RMLO)

        pos_encoding_LCC = PositionalEncoding(LCC)
        pos_encoding_RCC = PositionalEncoding(RCC)
        pos_encoding_LMLO = PositionalEncoding(LMLO)
        pos_encoding_RMLO = PositionalEncoding(RMLO)

        summer_LCC = pos_encoding_LCC.forward(LCC)
        summer_RCC = pos_encoding_RCC.forward(RCC)
        summer_LMLO = pos_encoding_LMLO.forward(LMLO)
        summer_RMLO = pos_encoding_RMLO.forward(RMLO)

        batched_positional_encoding = torch.zeros(
            batch_size, 50, 256)

        batched_positional_encoding[0] = summer_LCC
        batched_positional_encoding[1] = summer_LMLO
        batched_positional_encoding[2] = summer_RCC
        batched_positional_encoding[3] = summer_RMLO

        return batched_positional_encoding


# Global and Local mlp are equivalent
class MLP(nn.Module):
    def __init__(self, hidden_output=1024, dropout=.5):
        super(MLP, self).__init__()
        self.fnn1 = nn.Linear(256, hidden_output)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fnn2 = nn.Linear(hidden_output, 256)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, data):
        x = self.fnn1(data)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fnn2(x)
        x = self.gelu(x)
        x = self.dropout2(x)

        return x


class LocalEncoderBlock(nn.Module):
    def __init__(self, data_shape=(4, 50, 256), hidden_output_fnn1=1024, dropout=.5):
        super(LocalEncoderBlock, self).__init__()
        self.data_shape = data_shape
        # Layer norm over the H and W of each image
        self.ln1 = nn.LayerNorm([data_shape[1], data_shape[2]])
        self.ln2 = nn.LayerNorm([data_shape[1], data_shape[2]])

        self.attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=16, batch_first=True)
        self.mlp_0 = MLP(
            hidden_output=hidden_output_fnn1, dropout=dropout)
        self.mlp_1 = MLP(
            hidden_output=hidden_output_fnn1, dropout=dropout)
        self.mlp_2 = MLP(
            hidden_output=hidden_output_fnn1, dropout=dropout)
        self.mlp_3 = MLP(
            hidden_output=hidden_output_fnn1, dropout=dropout)

    def forward(self, data):
        if data.shape == (4, 256, 50):
            print('in here')
            data = data.T
        x = self.ln1(data)
        att_out, att_out_weights = self.attention(
            query=x, key=x, value=x)
        x_tilda = att_out + data
        x_second = self.ln2(x_tilda)
        dnn_output = torch.zeros(self.data_shape)
        dnn_output[0] = self.mlp_0.forward(x_second[0])
        dnn_output[1] = self.mlp_1.forward(x_second[1])
        dnn_output[2] = self.mlp_2.forward(x_second[2])
        dnn_output[3] = self.mlp_3.forward(x_second[3])
        x_second = dnn_output + x_tilda

        return x_second


class VisualTransformer(nn.Module):
    # embedding parameters, local encoder parameters
    def __init__(self, x_amount=7, y_amount=7, x_con=3500, y_con=2800,
                 data_shape=(4, 50, 256), hidden_output_fnn=1024, dropout=.5,
                 number_of_layers=10):
        super(VisualTransformer, self).__init__()
        self.embedding_block = EmbeddingBlock(
            x_amount=x_amount, y_amount=y_amount, x_con=x_con, y_con=y_con)
        self.blks = nn.Sequential()
        for i in range(number_of_layers):
            self.blks.add_module(
                f'{i}', LocalEncoderBlock(data_shape=data_shape))

    def forward(self, data):
        x = self.embedding_block.forward(data)
        print(x.shape)
        i = 0
        for blk in self.blks:
            print(f'This is {i} local attention run')
            i += 1
            x = blk(x)
        return x


class GlobalEncoderBlock(nn.Module):
    def __init__(self, data_shape=(1, 200, 256), hidden_output_fnn1=1024, dropout=.5):
        super(GlobalEncoderBlock, self).__init__()
        self.data_shape = data_shape
        self.gln1 = nn.LayerNorm(data_shape)
        self.ln2 = nn.LayerNorm(data_shape)
        self.attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=16, batch_first=True)
        self.mlp = MLP(hidden_output=hidden_output_fnn1, dropout=dropout)

    def forward(self, data):
        x = self.gln1(data)
        att_out, att_out_weights = self.attention(query=x, key=x, value=x)
        x_tilda = att_out + data
        x_second = self.ln2(x_tilda)
        dnn_output = self.mlp.forward(x_second)
        x_second = dnn_output + x_tilda

        return x_second


class GlobalTransformer(nn.Module):
    def __init__(self, x_amount=7, y_amount=7, x_con=3500, y_con=2800,
                 data_shape=(4, 50, 256), hidden_output_fnn=1024, dropout=.5,
                 number_of_layers=10, num_layers_global=10):
        super(GlobalTransformer, self).__init__()
        self.data_shape = data_shape
        new_data_shape = (1, data_shape[0]*data_shape[1], data_shape[2])
        self.individual_transformer = VisualTransformer(x_amount=x_amount, y_amount=y_amount, x_con=x_con,
                                                        y_con=y_con, data_shape=data_shape,
                                                        hidden_output_fnn=hidden_output_fnn,
                                                        dropout=dropout, number_of_layers=number_of_layers)
        self.blks = nn.Sequential()
        for i in range(num_layers_global):
            self.blks.add_module(
                f'{i}', GlobalEncoderBlock(data_shape=new_data_shape))

        self.flatten = nn.Flatten()

        # self.class_head = classification_head(input_layer=data_shape[0]*data_shape[2],
        #                                       hidden_output_class=512, dropout=.5)

    def forward(self, data):
        x = self.individual_transformer.forward(data)
        shape1, shape2, shape3 = x.shape
        x = torch.reshape(x, (1, shape1 * shape2, shape3))
        i = 0
        for blk in self.blks:
            print(f'This is {i} global attention run')
            x = blk(x)
            i += 1

        x = torch.squeeze(x)
        print(x.shape)
        x = x[[0, 1 * shape2, 2 * shape2, 3 * shape2], :]
        x = torch.reshape(x, (1, x.shape[0]*x.shape[1]))
        # x = class_head.forward(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_layer=1024, hidden_output_class=512, dropout=0.5):
        super(ClassificationHead, self).__init__()
        self.ln1 = nn.LayerNorm(input_layer)
        self.fnn1 = nn.Linear(input_layer, hidden_output_class)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_output_class)
        self.fnn2 = nn.Linear(hidden_output_class, 5)

    def forward(self, data):
        x = self.ln1(data)
        x = self.fnn1(x)
        x = self.dropout(x)
        x = self.ln2(x)
        x = self.fnn2(x)

        return x


class PaperModel(nn.Module):
    def __init__(self, x_amount=7, y_amount=7, x_con=3500, y_con=2800,
                 data_shape=(4, 50, 256), hidden_output_fnn=1024, dropout=.5,
                 number_of_layers=10, num_layers_global=10):

        super(PaperModel, self).__init__()

        self.embedding_block = EmbeddingBlock(
            x_amount, y_amount, x_con, y_con)

        self.visual_transformer = VisualTransformer(x_amount, y_amount, x_con, y_con,
                                                    data_shape, hidden_output_fnn, dropout,
                                                    number_of_layers)

        self.global_transformer = GlobalTransformer(x_amount, y_amount, x_con, y_con,
                                                    data_shape, hidden_output_fnn, dropout,
                                                    number_of_layers, num_layers_global)

        self.classification_head_left = ClassificationHead(
            input_layer=1024, hidden_output_class=512, dropout=0.5)

        self.classification_head_right = ClassificationHead(
            input_layer=1024, hidden_output_class=512, dropout=0.5)

    def forward(self, data):
        X = self.embedding_block(data)

        X = self.visual_transformer(X)

        X = self.global_transformer(X)

        left_pred = self.classification_head_left(X)

        right_pred = self.classification_head_right(X)

        return torch.vstack((left_pred, right_pred)).T
