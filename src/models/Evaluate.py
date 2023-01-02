import torch
from MyClasses import PaperModel
from data import get_train_test_dataloader
from runners import Tester
import boto3 as boto
from MyClasses import PaperModel

def load_model(model, model_path, name) -> Tester:

    s3_resource= boto.resource('s3')
    model_weights = torch.load(
        s3_resource.Bucket('mammographydata').download_file(key = model_path, Filename = name))
    
    model.load_state_dict(model_weights)

    tester = Tester(model = model)

    return tester

def test_model(tester, data):
    tester.evaluate(data, sv_roc = False)

if __name__ == '__main__':

    #TODO define data path for train/test examples
    data_path =''
    
    #TODO define path where these dictionaries will be saved from training 
    weight_path = ''

    #TODO define the name of weight dictionaries
    weight_name = ''

    split = (.8,.2)

    train_data, test_data  = get_train_test_dataloader(split = split, 
                            sequential = True, path = data_path, batch = 1)
    
    model = PaperModel(x_amount=7, y_amount=7, x_con=3500, y_con=2800,
                 data_shape=(4, 50, 256), hidden_output_fnn=1024, dropout=.5,
                 number_of_layers=10, num_layers_global=10)

    tester = load_model(model, weight_path, weight_name)
    test_model(tester, test_data)