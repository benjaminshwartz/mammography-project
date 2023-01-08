import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from datetime import datetime, timedelta


from data import get_train_test_dataloader
from MyClasses import PaperModel 
from runners import Trainer


       
def main(batch_size: int = 1, sequential : bool = False,split: tuple = (.8,.2), path :str = None):

    training_gen, test_gen = get_train_test_dataloader(split = split, sequential= sequential, path = path,batch=batch_size)

    model = PaperModel(x_amount=7, y_amount=7, x_con=3500, y_con=2800,
                 data_shape=(4, 50, 256), hidden_output_fnn=1024, dropout=.5,
                 number_of_layers=10, num_layers_global=10)

    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, weight_decay=0.0001)

    ce_loss = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model = model, optimizer=adam_optimizer, loss_fn=ce_loss, gpu_id='cuda', save_interval=50,
                      metric_interval=1, train_data=training_gen, validation_data=test_gen)


    s = datetime.now()
    print('Starting Training')
    num_epochs = 100
    trainer.train(num_epochs)
    print('Finished Training')
    f = datetime.now()
    print(f'Time to run {num_epochs} epochs: {f-s} (HH:MM:SS)')


if __name__ == "__main__":
    main()

