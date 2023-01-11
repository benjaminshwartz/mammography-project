import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pandas as pd
import pickle
import random
import itertools
import time
from datetime import datetime, timedelta
from torchmetrics import ROC
from torchmetrics.classification import BinaryROC
from matplotlib import pyplot as plt
import boto3 as boto
import torch.multiprocessing as mp

CHECKPOINT_PATH = ""


class Trainer():
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn,
        gpu_id: str,
        save_interval: int,
        metric_interval: int,
        train_data: DataLoader,
        validation_data: DataLoader = None,
        test_data: DataLoader = None
    ) -> None:
        ############ GPU RUNNING ########
        self.model = model.to(gpu_id)
        
        # self.model = model
        self.train_data = train_data
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gpu_id = gpu_id
        self.save_interval = save_interval
        self.metric_interval = metric_interval
        self.validation_data = validation_data
        self.test_data = test_data
        self.s3 = boto.client('s3')

    def _run_batch(self, batch_tensor: torch.tensor, batch_labels: torch.tensor):
        self.optimizer.zero_grad()
        
        predicted_output = self.model(batch_tensor).to(self.gpu_id)
        # print(f'SHAPE OF PREDICTED OUTPUT: {predicted_output.shape}')
        batch_labels = torch.reshape(batch_labels, (predicted_output.shape[0],2))
        # print(f'SHAPE OF LABELS: {batch_labels.shape}')
        loss = self.loss_fn(predicted_output, batch_labels.long())
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch: int):
        self.model.train()
        print(f'\t[GPU {self.gpu_id}] Epoch {epoch}')
        # i = 1
        # all = len(self.train_data)
        
        i = 0
        for batch_tensor, batch_labels in self.train_data:
            # print(f'\t{i}/{len(self.train_data)}')
            # i += 1
            
            ########## GPU RUNNING ##############
            batch_tensor = batch_tensor.to(self.gpu_id)
            batch_labels = batch_labels.to(self.gpu_id)
            
            self._run_batch(batch_tensor, batch_labels.float())
            i += 1
            if i % 20 == 0:
                print(f'Have run {i} batches')
            

    #TODO delete checkpoint file after upload to s3 bucket
    def _save_checkpoint(self, epoch: int):
        checkpoint = self.model.state_dict()
        #torch.save(checkpoint, CHECKPOINT_PATH)
        pickle.dump(checkpoint, open(f'checkpoint_{epoch}.pt','wb'))
        self.s3.upload_file(f'checkpoint_{epoch}.pt', 'mammographydata', f'DataSet/checkpointmodels/checkpoint_{epoch}.pt')
        print(f'\tModel Saved at Epoch {epoch}')

    def train(self, num_epochs: int, sv_roc: bool = False):
        self.model.share_memory()
        for epoch in range(1, num_epochs + 1):
            #Trying to multiprocess
            # processes = []
            # ctx = mp.get_context('spawn')
            # for i in range(os.cpu_count()):
            #     print(f'trying to make process {i}')
            #     p = ctx.Process(target=self._run_epoch, args=(epoch,))
            #     print(f'have made process {i}')
            #     p.start()
            #     processes.append(p)
            # print('out of loop')
            # for p in processes: 
            #     p.join()
            #No multiprocess
            
            self._run_epoch(epoch)

            if self.save_interval > 0 and epoch % self.save_interval == 0:
                self._save_checkpoint(epoch)
            elif epoch == num_epochs:  # save last model
                self._save_checkpoint(epoch)
            if self.metric_interval > 0 and epoch % self.metric_interval == 0:
                print("\tTrain Metrics (Training Data):")
                self.evaluate(self.train_data, sv_roc=sv_roc)
                if self.test_data != None:
                    print("\tTest Metrics:")
                    self.evaluate(self.test_data)
                    self.model.train()
            elif epoch == num_epochs:  # Evaluate final model
                print("\tTrain Metrics:")
                self.evaluate(self.train_data, sv_roc=sv_roc)
                # if self.validation_data != None:
                #     print("\tTest Metrics:")
                #     self.evaluate(self.validation_data)

    def evaluate(self, dataloader: DataLoader, sv_roc=False):

        with torch.no_grad():
            self.model.eval()
            cumulative_loss = 0
            left_cumulative_loss = 0
            right_cumulative_loss = 0
            num_correct = 0
            num_correct_left = 0
            num_correct_right = 0
            total = 0
            num_batches = len(dataloader)
            # all_preds = []  # torch.tensor([]).to(self.gpu_id)

            for batch_tensor, batch_labels in dataloader:
                
                #######GPU RUNNING#####
                batch_tensor = batch_tensor.to(self.gpu_id)
                # we want batch_labels.shape = B, 5, 2
                # we want predicted_output = B, 5, 2
                batch_labels = batch_labels.to(self.gpu_id).long()
                

                predicted_output = self.model(batch_tensor).to(self.gpu_id)
                left_preds = predicted_output[:,:,0].to(self.gpu_id)
                right_preds = predicted_output[:,:,1].to(self.gpu_id)
                
                batch_labels = batch_labels.long()
                batch_labels = torch.reshape(batch_labels, (predicted_output.shape[0],2)).to(self.gpu_id)
                left_labels = batch_labels[:,0].to(self.gpu_id)
                right_labels = batch_labels[:,1].to(self.gpu_id)

                cumulative_loss += self.loss_fn(predicted_output, batch_labels)
                left_cumulative_loss += self.loss_fn(left_preds, left_labels)
                right_cumulative_loss += self.loss_fn(
                    right_preds, right_labels)

                if sv_roc:
                    # TODO fix roc curve
                    pass
                else:
                    pass
                # print(f'THIS IS THE RIGHT LABEL: {right_labels}')
                print(f'THIS IS THE RIGHT PREDICTED LABEL: {right_preds}')
                # print('##################################')
                # print(f'THIS IS THE LEFT LABEL: {left_labels}')
                print(f'THIS IS THE LEFT PREDICTED LABEL: {left_preds}')
                # print('++++++++++++++++++++++++++++++++++++++++++')
                total += len(left_labels)
                print(f'TOTAL NUMBER: {total}')
                
                # num_correct_left += (torch.argmax(left_preds, dim=0)
                #                      == torch.argmax(left_labels, dim=0)).sum().item()
                
                # num_correct_right += (torch.argmax(right_preds, dim=0)
                #                       == torch.argmax(right_labels, dim=0)).sum().item()
                left_positions = torch.argmax(left_preds, dim=1)
                right_positions = torch.argmax(right_preds, dim=1)
                print(f'LEFT PREDS: {left_positions}')
                print(f'RIGHT PREDS: {right_positions}')
                print('-------------------------------------')
                print(f'LEFT LABELS: {left_labels}')
                print(f'RIGHT LABELS: {right_labels}')
                print('--------------------------------------')
                
                num_correct_left += (left_positions == left_labels).sum().item()
                
                print(f'NUMBER CORRECT STATED LEFT: {num_correct_left}')

                
                num_correct_right += (right_positions == right_labels).sum().item()
                
                print(f'NUMBER CORRECT STATED RIGHT: {num_correct_right}')
                print('##################################')
                
                for i in range(len(left_positions)):
                    if (left_positions[i] == left_labels[i]) and (right_positions[i] == right_labels[i]):
                        num_correct += 1
                
                # num_correct += ((torch.argmax(right_preds, dim=0) == right_labels) and (torch.argmax(left_preds, dim=0))).sum().item()
                
                print(f'TOTAL NUM CORRECT: {num_correct}')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            loss = cumulative_loss/num_batches
            left_loss = left_cumulative_loss / num_batches
            right_loss = right_cumulative_loss / num_batches
            accuracy = num_correct/total
            accuracy_left = num_correct_left/total
            accuracy_right = num_correct_right/total

            print(
                f'\t\tOverall Loss: {loss} = {cumulative_loss}/{num_batches}')
            print(
                f'\t\tLeft Loss: {left_loss} = {left_cumulative_loss}/{num_batches}')
            print(
                f'\t\tRight Loss: {right_loss} = {right_cumulative_loss}/{num_batches}')

            print(f'\t\tOverall Accuracy: {accuracy} = {num_correct}/{total}')
            print(
                f'\t\tLeft Accuracy: {accuracy_left} = {num_correct_left}/{total}')
            print(
                f'\t\tRight Accuracy: {accuracy_right} = {num_correct_right}/{total}')

            if sv_roc:
                # TODO fix save roc
                # Trainer.save_roc(all_preds, all_labels)
                pass

        self.model.train()

    class Tester:
        def __init__(
            self,
            model: torch.nn.Module,
            loss_fn: torch.nn = None,
            gpu_id: int = 0,
        ) -> None:
            self.loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
            self.gpu_id = gpu_id
            # self.model = model.to(self.gpu_id)
            self.model = model

        def evaluate(self, dataloader: DataLoader, sv_roc=False):
            with torch.no_grad():
                self.model.eval()
                cumulative_loss = 0
                num_correct = 0
                total = 0
                num_batches = len(dataloader)
                # all_preds = torch.tensor([]).to(self.gpu_id)
                # all_labels = torch.tensor([]).to(self.gpu_id)
                all_preds = torch.tensor([])
                all_labels = torch.tensor([])

                for batch_tensor, batch_labels in dataloader:
                    
                    # batch_tensor = batch_tensor.to(self.gpu_id)
                    batch_tensor = batch_tensor
                    # check batch labels type
                    # batch_tensor = batch_tensor.to(self.gpu_id)
                # we want batch_labels.shape = B, 5, 2
                # we want predicted_output = B, 5, 2
                # batch_labels = batch_labels.to(self.gpu_id).long()
                batch_labels = batch_labels.long()
                batch_labels = torch.reshape(batch_labels, (2,))
                left_labels = batch_labels[0]
                right_labels = batch_labels[1]

                predicted_output = self.model(batch_tensor)
                left_preds = predicted_output[0]
                right_preds = predicted_output[1]

                cumulative_loss += self.loss_fn(predicted_output, batch_labels)
                left_cumulative_loss += self.loss_fn(left_preds, left_labels)
                right_cumulative_loss += self.loss_fn(
                    right_preds, right_labels)
                
                if sv_roc:
                    softmax = nn.Softmax(dim=1)
                    all_preds = torch.cat(
                        (all_preds, (softmax(predicted_output)[:, 1])))
                    all_labels = torch.cat((all_labels, batch_labels))

                # assuming decision boundary to be 0.5
                total += batch_labels.size(0)
                
#                 num_correct_left += (torch.argmax(left_preds, dim=0)
#                                      == torch.argmax(left_labels, dim=0)).sum().item()

#                 num_correct_right += (torch.argmax(right_preds, dim=0)
#                                       == torch.argmax(right_labels, dim=0)).sum().item()


            num_correct_left += (torch.argmax(left_preds, dim=0) == left_labels).sum().item()

                
            num_correct_right += (torch.argmax(right_preds, dim=0) == right_labels).sum().item()

            loss = cumulative_loss/num_batches
            left_loss = left_cumulative_loss / num_batches
            right_loss = right_cumulative_loss / num_batches
            accuracy = num_correct/total
            half = total/2
            accuracy_left = num_correct_left/half
            accuracy_right = num_correct_right/half

            print(
                f'\t\tOverall Loss: {loss} = {cumulative_loss}/{num_batches}')
            print(
                f'\t\tLeft Loss: {left_loss} = {left_cumulative_loss}/{num_batches}')
            print(
                f'\t\tRight Loss: {right_loss} = {right_cumulative_loss}/{num_batches}')

            print(f'\t\tAccuracy: {accuracy} = {num_correct}/{total}')
            print(
                f'\t\tLeft Accuracy: {accuracy_left} = {num_correct_left}/{half}')
            print(
                f'\t\tRight Accuracy: {accuracy_right} = {num_correct_right}/{half}')

    # TODO: fix for (B,2,5) prediction
    @ staticmethod
    def save_roc(all_preds, all_labels):
        roc = ROC(task="binary", thresholds=20)
        roc = BinaryROC(thresholds=100)
        all_preds = all_preds.cpu()
        all_labels = all_labels.cpu().int()
        fpr, tpr, thresholds = roc(all_preds, all_labels)
        plt.plot([0, 1], [0, 1], linestyle='dashed')
        plt.plot(fpr, tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('ROC.png')
