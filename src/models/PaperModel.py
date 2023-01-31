import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from datetime import datetime, timedelta


from data import get_train_test_dataloader
# from mammography_project.src.models.MyClasses import PaperModel 
# from mammography_project.src.models.BatchedMyClasses import PaperModel 
# from mammography_project.src.models.runners import Trainer
from BatchedMyClasses import PaperModel 
from runners import Trainer


       
def main(batch_size: int = 1,device: str = 'cpu', sequential : bool = False,split: tuple = (.8,.2), path :str = None):

    training_gen, test_gen = get_train_test_dataloader(split = split, sequential= sequential, batch =batch_size)
    print('Trying Batched')
    model = PaperModel(x_amount=7, y_amount=7, x_con=3500, y_con=2800,
                 data_shape=(batch_size, 4, 50, 256), hidden_output_fnn=1024, dropout=.5,
                 number_of_layers=10, num_layers_global=10)
    
    model = model.to(device)

    # adam_optimizer = torch.optim.Adam(
    #     model.parameters(), lr=0.0001, weight_decay=0.0001)
    
    adagrad_optimizer = torch.optim.Adagrad(
        model.parameters(), lr=0.001, weight_decay=0.0001)

    ce_loss = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model = model, optimizer=adagrad_optimizer, loss_fn=ce_loss, gpu_id='cuda', save_interval=10,
                      metric_interval=1, train_data=training_gen, test_data=test_gen)


    s = datetime.now()
    print('Starting Training')
    num_epochs = 1000
    trainer.train(num_epochs)
    print('Finished Training')
    f = datetime.now()
    print(f'Time to run {num_epochs} epochs: {f-s} (HH:MM:SS)')


if __name__ == "__main__":
    main(batch_size = 2, device = 'cuda')

