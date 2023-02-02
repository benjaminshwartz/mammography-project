import torch
from torch import nn
import numpy as np
import pandas as pd
import os
import pydicom as dicom
import math
import time


class PositionalEncoding(nn.Module):
    def __init__(self, data, dropout=0.1, n=10000):
        super(PositionalEncoding, self).__init__()
        self.batch_size, self.embedded_dim, self.position = data.shape
        self.dropout = nn.Dropout(p=dropout)
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'

        self.embedded_dim += 1  # adding one to embedded dim to take into account token prepend

        self.learned_embedding_vec = nn.Parameter(
            torch.zeros(self.batch_size, 1, self.position)).to(device)

        self.positional_matrix = torch.zeros(
            self.embedded_dim, self.position).to(device)

        for pos in range(self.position):
            for i in range(int(self.embedded_dim/2)):
                denom = pow(n, 2*i/self.embedded_dim)
                self.positional_matrix[2*i, pos] = np.sin(pos/denom)
                self.positional_matrix[2*i+1, pos] = np.cos(pos/denom)

        self.positional_matrix = self.positional_matrix[None, :, :]
        self.positional_matrix = self.positional_matrix.tile(
            (self.batch_size, 1, 1))

    def forward(self, data):
        data = torch.hstack((self.learned_embedding_vec, data))
        summer_matrix = data + self.positional_matrix
        summer_matrix = self.dropout(summer_matrix)

        return summer_matrix


class ConvLayer(nn.Module):

    def __init__(self, batch_size, num_patch: int = 49):
        super(ConvLayer, self).__init__()

        if torch.cuda.is_available():
            device = 'cuda'

        self.num_patch = num_patch
        self.batch_size = batch_size
        n = num_patch
#         self.conv2d_1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 13, stride = 1)
        self.conv2d_1 = nn.Conv2d(
            in_channels=n*1, out_channels=n * 8, kernel_size=13, stride=1, groups=n)

        self.pooling2d_1 = nn.MaxPool2d(2)

        self.conv2d_2 = nn.Conv2d(
            in_channels=n*8, out_channels=n*16, kernel_size=11, stride=1, groups=n)

        self.pooling2d_2 = nn.MaxPool2d(2)

#         self.conv2d_3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 9, stride = 1, groups = n)
        self.conv2d_3 = nn.Conv2d(
            in_channels=n*16, out_channels=n*32, kernel_size=9, stride=1, groups=n)


#         self.conv2d_4 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 7, stride = 1, groups = n)
        self.conv2d_4 = nn.Conv2d(
            in_channels=n*32, out_channels=n*32, kernel_size=7, stride=1, groups=n)

        self.pooling2d_3 = nn.MaxPool2d(2)

#         self.conv2d_5 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, groups = n)
        self.conv2d_5 = nn.Conv2d(
            in_channels=n*32, out_channels=n*64, kernel_size=5, stride=1, groups=n)

        self.dnn = nn.Linear(105280, 256)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, tensor):

        # print('IN FORWARD OF CONV LAYER')
        tensor = tensor[:,]
        # print(f'THIS IS THE SHAPE OF THE TENSOR: {tensor.shape}')
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
        # print(x.shape)
        x = torch.reshape(x, (self.batch_size, self.num_patch, 105280))
        x = self.dnn(x)

        return x


class EmbeddingBlock(nn.Module):
    # Data in this sense is the image that has not been translated into an array
    # Want to set x_con to 3500
    def __init__(self, batch, x_amount=7, y_amount=7, x_con=3500, y_con=2800):
        super(EmbeddingBlock, self).__init__()

        assert (x_con % x_amount == 0)
#         print(y_con)
#         print(y_amount)
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

        self.cc_conv = ConvLayer(batch_size=batch)
        self.mlo_conv = ConvLayer(batch_size=batch)

    def forward(self, data):
        # recheck for proper class variables(change self. to strictly local variable)
        # print('IN FORWARD OF EMBEDDING BLOCK LAYER')
        # Data shape (batch size, num of views, x_length, y_length)
        info = data

        batch_size = info.shape[0]

        batched_patches = info.unfold(
            2, self.x_ran, self.x_ran).unfold(3, self.y_ran, self.y_ran)
        batched_patches = torch.reshape(batched_patches,
                                        (batch_size, 4, self.amount_of_patches, self.x_ran, self.y_ran))

        # Reshape now makes data (batch_size,4,49,500,400)

        batched_positional_encoding = torch.zeros(batch_size, 4, 50, 256)

        LCC = batched_patches[:, 0]
        LMLO = batched_patches[:, 1]
        RCC = batched_patches[:, 2]
        RMLO = batched_patches[:, 3]

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

        batched_positional_encoding[:, 0] = summer_LCC
        batched_positional_encoding[:, 1] = summer_LMLO
        batched_positional_encoding[:, 2] = summer_RCC
        batched_positional_encoding[:, 3] = summer_RMLO

        return batched_positional_encoding


class MLP(nn.Module):
    def __init__(self, hidden_output=1024, dropout=.5):
        super(MLP, self).__init__()
        self.fnn1 = nn.Linear(256, hidden_output)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fnn2 = nn.Linear(hidden_output, 256)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, data):
        # print('IN FORWARD OF MLP LAYER')
        x = self.fnn1(data)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fnn2(x)
        x = self.gelu(x)
        x = self.dropout2(x)

        return x


class LocalEncoderBlock(nn.Module):
    def __init__(self, data_shape, hidden_output_fnn1=1024, dropout=.5):
        super(LocalEncoderBlock, self).__init__()
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.data_shape = data_shape
        # Layer norm over the H and W of each image
        self.batch_size = 10
#         print([data_shape[2], data_shape[3]])
        self.ln1 = nn.LayerNorm(
            [data_shape[2], data_shape[3]], device=self.device)
        self.ln2 = nn.LayerNorm(
            [data_shape[2], data_shape[3]], device=self.device)

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

        data.to(self.device)
        x_tilda_matrix = torch.zeros(self.data_shape)
        attn_0, y = self.helper_thing(data[:, 0])
        x_tilda_matrix[:, 0] = y
        attn_1, y = self.helper_thing(data[:, 1])
        x_tilda_matrix[:, 1] = y
        attn_2, y = self.helper_thing(data[:, 2])
        x_tilda_matrix[:, 2] = y
        attn_3, y = self.helper_thing(data[:, 3])
        x_tilda_matrix[:, 3] = y

        dnn_output = torch.zeros(self.data_shape)
        dnn_output[:, 0] = self.mlp_0.forward(attn_0)
        dnn_output[:, 1] = self.mlp_1.forward(attn_1)
        dnn_output[:, 2] = self.mlp_2.forward(attn_2)
        dnn_output[:, 3] = self.mlp_3.forward(attn_3)
        x_second = dnn_output + x_tilda_matrix

        return x_second

    def helper_thing(self, data):
        # Data should be of shape (batch_size, 256, 50)

        data = data.to(self.device)
        # print(self.device)
        # print(f'Data Device:{data.device}')
        # print(f'ln1 device: {self.ln1.device}')
        x = self.ln1(data)
#         print(f'x.shape: {x.shape}')
        att_out, att_out_weights = self.attention(
            query=x, key=x, value=x)
        att_out = att_out.to(self.device)
        x_tilda = att_out + data
        x_second = self.ln2(x_tilda)
#         print(x_tilda.shape)

        return x_second, x_tilda


class VisualTransformer(nn.Module):
    # embedding parameters, local encoder parameters
    def __init__(self, x_amount=7, y_amount=7, x_con=3500, y_con=2800,
                 data_shape=(10, 4, 50, 256), hidden_output_fnn=1024, dropout=.5,
                 number_of_layers=10):
        super(VisualTransformer, self).__init__()
        self.embedding_block = EmbeddingBlock(batch=data_shape[0],
                                              x_amount=x_amount, y_amount=y_amount, x_con=x_con, y_con=y_con)
        self.blks = nn.Sequential()
        for i in range(number_of_layers):
            self.blks.add_module(
                f'{i}', LocalEncoderBlock(data_shape=data_shape))

    def forward(self, data):
        # print('IN FORWARD OF VISUALTRANSFORMER LAYER')
        x = self.embedding_block.forward(data)
        # print(x.shape)
        i = 0
        for blk in self.blks:
            # print(f'This is {i} local attention run')
            i += 1
            x = blk(x)
        return x


class GlobalEncoderBlock(nn.Module):
    def __init__(self, data_shape=(10, 200, 256), hidden_output_fnn1=1024, dropout=.5):
        super(GlobalEncoderBlock, self).__init__()
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.data_shape = data_shape
        self.gln1 = nn.LayerNorm(data_shape, device=self.device)
        self.ln2 = nn.LayerNorm(data_shape, device=self.device)
        self.attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=16, batch_first=True)
        self.mlp = MLP(hidden_output=hidden_output_fnn1, dropout=dropout)

    def forward(self, data):
        # print('IN FORWARD OF GLOBALENCODERBLOCK LAYER')
        data = data.to(self.device)
        x = self.gln1(data)
        att_out, att_out_weights = self.attention(query=x, key=x, value=x)
        att_out = att_out.to(self.device)
        x_tilda = att_out + data
        x_second = self.ln2(x_tilda)
        dnn_output = self.mlp.forward(x_second)
        x_second = dnn_output + x_tilda

        return x_second


class GlobalTransformer(nn.Module):
    def __init__(self, x_amount=7, y_amount=7, x_con=3500, y_con=2800,
                 data_shape=(10, 4, 50, 256), hidden_output_fnn=1024, dropout=.5,
                 number_of_layers=10, num_layers_global=10):
        super(GlobalTransformer, self).__init__()
        self.data_shape = data_shape
        new_data_shape = (data_shape[0], data_shape[1]
                          * data_shape[2], data_shape[3])
        self.blks = nn.Sequential()
        for i in range(num_layers_global):
            self.blks.add_module(
                f'{i}', GlobalEncoderBlock(data_shape=new_data_shape))

        self.flatten = nn.Flatten()

        # self.class_head = classification_head(input_layer=data_shape[0]*data_shape[2],
        #                                       hidden_output_class=512, dropout=.5)

    def forward(self, data):
        # print('IN FORWARD OF GLOBALTRANSFORMER LAYER')
        #x = self.individual_transformer.forward(data)

        shape0, shape1, shape2, shape3 = data.shape
        x = torch.reshape(data, (shape0, shape1 * shape2, shape3))
        i = 0
        for blk in self.blks:
            # print(f'This is {i} global attention run')
            x = blk(x)
            i += 1

#         x = torch.squeeze(x)
        # print(x.shape)
        x = x[:, [0, 1 * shape2, 2 * shape2, 3 * shape2], :]
        x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
#         print(x.shape)
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
        # print('IN FORWARD OF CLASSIFICATIONHEAD LAYER')
        x = self.ln1(data)
        x = self.fnn1(x)
        x = self.dropout(x)
        x = self.ln2(x)
        x = self.fnn2(x)

        return x


class RegressionHead(nn.Module):
    def __init__(self, input_layer=1024, hidden_output_class=512, dropout=0.5):
        super(ClassificationHead, self).__init__()
        self.ln1 = nn.LayerNorm(input_layer)
        self.fnn1 = nn.Linear(input_layer, hidden_output_class)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_output_class)
        self.fnn2 = nn.Linear(hidden_output_class, 1)

    def forward(self, data):
        # print('IN FORWARD OF REGRESSIONHEAD LAYER')
        x = self.ln1(data)
        x = self.fnn1(x)
        x = self.dropout(x)
        x = self.ln2(x)
        x = self.fnn2(x)

        return x


class PaperModel(nn.Module):
    def __init__(self, x_amount=7, y_amount=7, x_con=3500, y_con=2800,
                 data_shape=(10, 4, 50, 256), hidden_output_fnn=1024, dropout=.5,
                 number_of_layers=10, num_layers_global=10, setting='C'):

        assert setting in {'C', 'R'}

        super(PaperModel, self).__init__()

        self.embedding_block = EmbeddingBlock(batch=data_shape[0],
                                              x_amount=x_amount, y_amount=y_amount, x_con=x_con, y_con=y_con)

        self.visual_transformer = VisualTransformer(x_amount, y_amount, x_con, y_con,
                                                    data_shape, hidden_output_fnn, dropout,
                                                    number_of_layers)

        self.global_transformer = GlobalTransformer(x_amount, y_amount, x_con, y_con,
                                                    data_shape, hidden_output_fnn, dropout,
                                                    number_of_layers, num_layers_global)

        if setting == 'C':

            self.left_head = ClassificationHead(
                input_layer=1024, hidden_output_class=512, dropout=0.5)

            self.right_head = ClassificationHead(
                input_layer=1024, hidden_output_class=512, dropout=0.5)

        elif setting == 'R':
            self.left_head = RegressionHead(
                input_layer=1024, hidden_output_class=512, dropout=0.5)

            self.right_head = RegressionHead(
                input_layer=1024, hidden_output_class=512, dropout=0.5)

    def forward(self, data):
        #X = self.embedding_block(data)
        batch = data.shape[0]

        #data = torch.reshape(data, (4,3500,2800))
        # here
        # print(f'THIS IS THE DATA SHAPE: {data.shape}')

        X = self.visual_transformer(data)

        X = self.global_transformer(X)

        left_pred = self.left_head(X)

        # print(f'left shape: {left_pred.shape}')

        right_pred = self.right_head(X)
        # print(f'right shape: {right_pred.shape}')

        # final = torch.zeros(left_pred.shape[0], 5, 2)
        # final[:, :, 0] = left_pred
        # final[:, :, 1] = right_pred

        final = torch.stack((left_pred, right_pred), dim=2)

        # print(f'Finished data classification, returning vectors of shape: {final.shape}')

        return final
