# Torch Stuff
import torch
import torchvision
import torchvision.models as models
from torch import nn, autograd as ag
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as init

from helpers.ops import shuffle_unison
from tasks import TaskGen

import os, joblib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from models import LeNet

import argparse
seed = 42
rng = np.random.RandomState(seed)
torch.manual_seed(seed)


class Reptile(object):
    def __init__(self, args):
        self.args = args
        self._load_model()

        self.model.to(args.device)
        self.task_generator = TaskGen(args.max_num_classes)
        self.outer_stepsize = args.outer_stepsize
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=args.inner_stepsize)

    def _load_model(self):
        self.model = LeNet()
        self.current_iteration = 0
        if os.path.exists(self.args.model_path):
            try:
                print("Loading model from: {}".format(self.args.model_path))
                self.model.load_state_dict(torch.load(self.args.model_path))
                self.current_iteration = joblib.load("{}.iter".format(self.args.model_path))
            except Exception as e:
                print("Exception: {}\nCould not load model from {} - starting from scratch".format(e, self.args.model_path))

    def inner_training(self, x, y, num_iterations):
        """
        Run training on task
        """
        x, y = shuffle_unison(x, y)

        self.model.train()
        
        x = torch.tensor(x, dtype=torch.float, device=self.args.device)
        y = torch.tensor(y, dtype=torch.float, device=self.args.device)

        total_loss = 0
        for _ in range(num_iterations):
            start = np.random.randint(0, len(x)-self.args.inner_batch_size+1)
           
            self.model.zero_grad()
            # self.optimizer.zero_grad()
            outputs = self.model(x[start:start+self.args.inner_batch_size])
            # print("output: {} - y: {}".format(outputs.shape, y.shape))
            loss = self.criterion(outputs, Variable(y[start:start+self.args.inner_batch_size].long()))
            total_loss += loss
            loss.backward()
            # self.optimizer.step()
            # Similar to calling optimizer.step()
            for param in self.model.parameters():
                param.data -= self.args.inner_stepsize * param.grad.data
        return total_loss / self.args.inner_iterations

    def _meta_gradient_update(self, iteration, num_classes, weights_before):
        """
        Interpolate between current weights and trained weights from this task
        I.e. (weights_before - weights_after) is the meta-gradient

            - iteration: current iteration - used for updating outer_stepsize
            - num_classes: current classifier number of classes
            - weights_before: state of weights before inner steps training
        """
        weights_after = self.model.state_dict()
        outer_stepsize = self.outer_stepsize * (1 - iteration / self.args.n_iterations) # linear schedule
        
        self.model.load_state_dict({name :
            weights_before[name] + (weights_after[name] - weights_before[name]) * outer_stepsize
            for name in weights_before})
        
    def meta_training(self):
        # Reptile training loop
        total_loss = 0
        try:
            while self.current_iteration < self.args.n_iterations:
                # Generate task
                data, labels, original_labels, num_classes = self.task_generator.get_train_task(args.num_classes)
                
                weights_before = deepcopy(self.model.state_dict())
                loss = self.inner_training(data, labels, self.args.inner_iterations)
                total_loss += loss
                if self.current_iteration % self.args.log_every == 0:
                    print("-----------------------------")
                    print("iteration               {}".format(self.current_iteration+1))
                    print("Loss: {:.3f}".format(total_loss/(self.current_iteration+1)))
                    print("Current task info: ")
                    print("\t- Number of classes: {}".format(num_classes))
                    print("\t- Batch size: {}".format(len(data)))
                    print("\t- Labels: {}".format(set(original_labels)))
 
                    self.test()
                
                self._meta_gradient_update(self.current_iteration, num_classes, weights_before)
                
                self.current_iteration += 1

            torch.save(self.model.state_dict(), self.args.model_path)                    

        except KeyboardInterrupt:
            print("Manual Interrupt...")
            print("Saving to: {}".format(self.args.model_path))
            torch.save(self.model.state_dict(), self.args.model_path)
            joblib.dump(self.current_iteration, "{}.iter".format(self.args.model_path), compress=1)

    def predict(self, x):
        self.model.eval()
        x = torch.tensor(x, dtype=torch.float, device=self.args.device)
        outputs = self.model(x)
        return outputs.cpu().data.numpy()

    def test(self):
        """
        Run tests
            1. Create task from test set.
            2. Reload model
            3. Check accuracy on test set
            4. Train for one or more iterations on one task
            5. Check accuracy again on test set
        """
        
        test_data, test_labels, _, _ = self.task_generator.get_test_task(selected_labels=[1,2,3,4,5], num_samples=-1) # all available samples
        predicted_labels = np.argmax(self.predict(test_data), axis=1)
        accuracy = np.mean(1*(predicted_labels==test_labels))*100
        print("Accuracy before few shots learning (a.k.a. zero-shot learning): {:.2f}%\n----".format(accuracy))
    
        weights_before = deepcopy(self.model.state_dict()) # save snapshot before evaluation
        for i in range(1, 5):
            enroll_data, enroll_labels, _, _ = self.task_generator.get_enroll_task(selected_labels=[1,2,3,4,5], num_samples=i)
            self.inner_training(enroll_data, enroll_labels, self.args.inner_iterations_test)
            predicted_labels = np.argmax(self.predict(test_data), axis=1)
            accuracy = np.mean(1*(predicted_labels==test_labels))*100
            
            print("Accuracy after {} shot{} learning: {:.2f}%)".format(i, "" if i == 1 else "s", accuracy))

        self.model.load_state_dict(weights_before) # restore from snapshot
            
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Meta Learning for MNIST')
    parser.add_argument('--inner-stepsize', type=float, default=1E-3,
                        help='stepsize in inner optimizer')
    parser.add_argument('--inner-iterations', type=int, default=5,
                        help='Number of iterations inner training')
    parser.add_argument('--inner-iterations-test', type=int, default=5,
                        help='Number of iterations inner training at test time')
    parser.add_argument('--inner-batch-size', type=int, default=5,
                        help='Inner Batch Size')
    parser.add_argument('--outer-stepsize', type=float, default=1E-1,
                        help='stepsize of outer optimization, i.e., meta-optimization')
    parser.add_argument('--n-iterations', type=int, default=400000,
                        help='number of outer updates; each iteration we sample one task and update on it')
    parser.add_argument('--max-num-classes', type=int, default=9,
                        help='Max number of classes in the training set')
    parser.add_argument('--device', type=str, default='cpu',
                        help='HW acceleration')
    parser.add_argument('--log-every', type=int, default=2000,
                        help="Show progress every n iterations")
    parser.add_argument('--model-path', type=str, default='model.bin',
                        help='Path to were to save trained model')
    # parser.add_argument('--num-test-classes', type=int, default=5,
    #                     help='Number of classes for test tasks')
    parser.add_argument('--num-classes', type=int, default=5,
                    help='Number of classes for training tasks')

    args = parser.parse_args()

    task_generator = TaskGen(args.max_num_classes)
    reptile = Reptile(args)
    reptile.meta_training()
    reptile.test()
