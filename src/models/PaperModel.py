import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from datetime import datetime, timedelta


from data import get_train_test_dataloader
# from mammography_project.src.models.MyClasses import PaperModel
# from mammography_project.src.models.BatchedMyClasses import PaperModel
# from mammography_project.src.models.runners import Trainer
from BatchedMyCLasses import PaperModel
from runners import Trainer
import torch.multiprocessing as mp

from torch.distributed import init_process_group, destroy_process_group
import os
from contextlib import closing
import socket


def get_open_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def ddp_setup(rank, world_size):
    # VERY UNSURE ABOUT THIS ASSIGNMENTS
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ["MASTER_PORT"] = str(get_open_port())
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank: int, world_size: int, batch_size: int = 1,
         device: str = 'cpu', sequential: bool = False, split: tuple = (.8, .2), path: str = None):

    ddp_setup(rank, world_size)
    training_gen, test_gen = get_train_test_dataloader(
        split=split, sequential=sequential, batch=batch_size)
    print('Trying Batched')
    #### Classification ####
    # model = PaperModel(x_amount=7, y_amount=7, x_con=3500, y_con=2800,
    #                    data_shape=(batch_size, 4, 50, 256), hidden_output_fnn=1024, dropout=.5,
    #                    number_of_layers=10, num_layers_global=10, setting='C')

    #### Regression ####
    model = PaperModel(x_amount=7, y_amount=7, x_con=3500, y_con=2800,
                       data_shape=(batch_size, 4, 50, 256), hidden_output_fnn=1024, dropout=.5,
                       number_of_layers=10, num_layers_global=10, setting='R')

    model = model.to(device)

    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, weight_decay=0.0001)

    # adagrad_optimizer = torch.optim.Adagrad(
    #     model.parameters(), lr=0.001, weight_decay=0.0001)

    ce_loss = torch.nn.CrossEntropyLoss()
    # mse_loss = torch.nn.MSELoss()

    trainer = Trainer(model=model, optimizer=adam_optimizer, loss_fn=ce_loss, gpu_id=rank, save_interval=10,
                      metric_interval=1, train_data=training_gen, test_data=test_gen)

    s = datetime.now()
    print('Starting Training')
    num_epochs = 1000
    trainer.train(num_epochs)
    destroy_process_group()
    print('Finished Training')
    f = datetime.now()
    print(f'Time to run {num_epochs} epochs: {f-s} (HH:MM:SS)')


def single_main(batch_size: int = 1, device: str = 'cpu', sequential: bool = False, split: tuple = (.8, .2), path: str = None):

    training_gen, test_gen = get_train_test_dataloader(
        split=split, sequential=sequential, batch=batch_size)
    print('Trying Batched')
    model = PaperModel(x_amount=7, y_amount=7, x_con=3500, y_con=2800,
                       data_shape=(batch_size, 4, 50, 256), hidden_output_fnn=1024, dropout=.5,
                       number_of_layers=10, num_layers_global=10)

    model = model.to(device)

    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, weight_decay=0.0001)

    # adagrad_optimizer = torch.optim.Adagrad(
    #     model.parameters(), lr=0.001, weight_decay=0.0001)

    ce_loss = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model=model, optimizer=adam_optimizer, loss_fn=ce_loss, gpu_id=device, save_interval=10,
                      metric_interval=1, train_data=training_gen, test_data=test_gen)

    s = datetime.now()
    print('Starting Training')
    num_epochs = 1000
    trainer.train(num_epochs)
    destroy_process_group()
    print('Finished Training')
    f = datetime.now()
    print(f'Time to run {num_epochs} epochs: {f-s} (HH:MM:SS)')


if __name__ == "__main__":

    ##### MULTIGPU RUN ###########
    print('trying to run')
    batch_size = 1
    device = 'cpu'
    sequential = False
    split = (.8, .2)
    path = None
    world_size = torch.cuda.device_count()
    print(f'world size {world_size}')
    mp.spawn(main, args=(world_size, batch_size,
             device, sequential, split, path))
    print('done spawning')

    ###### SINGLE GPU RUN ########
    # main(batch_size = 2, device = 'cuda')
