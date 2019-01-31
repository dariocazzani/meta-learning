import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
from torch import nn, autograd as ag
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as init

import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from models import LeNet
from sklearn import preprocessing

import argparse
seed = 42
random.seed(seed)
rng = np.random.RandomState(seed)
torch.manual_seed(seed)

def shuffle_unison(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a, b

class TaskGen(object):
    def __init__(self, max_num_classes):
        self._transform = transform=torchvision.transforms.Compose([
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
        ])

        self.max_num_classes = max_num_classes

        self.sample_data_size = 1024
        self.max_samples_pre_class = 10
        self.trainset = datasets.MNIST(root='./data',
                                        train=True,
                                        download=True,
                                        transform=self._transform)
        self.testset = datasets.MNIST(root='./data',
                                        train=False,
                                        download=True,
                                        transform=self._transform)

        self.train_loader = torch.utils.data.DataLoader(self.trainset,
                                            batch_size=self.sample_data_size,
                                            shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.testset,
                                            batch_size=10000,
                                            shuffle=True)
        self.label_encoder = preprocessing.LabelEncoder()

    def _get_task(self, data_loader, num_classes, selected_labels, num_samples):
        """
            - data_loader
            - num_classes
            - selected_labels
            - num_samples
        """
        if len(selected_labels) == 0:
            selected_labels = random.sample(range(0, self.max_num_classes), num_classes)
        if len(selected_labels) != num_classes:
            raise ValueError("The number of selected labels and num classes is not the same")

        _, (data, labels) = next(enumerate(data_loader))
        data = data.numpy()
        labels = labels.numpy()

        preprocessed_labels = []
        preprocessed_data = []
        for c in range(num_classes):
            if num_samples == 0:
                class_num_samples = random.randint(2, self.max_samples_pre_class)
            else:
                class_num_samples = num_samples

            avail_data_idx = np.where(labels==selected_labels[c])[0]
            random.shuffle(avail_data_idx)
            avail_data_idx = avail_data_idx[:class_num_samples]
            for d in data[avail_data_idx]:
                preprocessed_data.append(d)
            for l in labels[avail_data_idx]:
                preprocessed_labels.append(l)

        # Shuffle in unison
        for _ in range(10):
            preprocessed_data, preprocessed_labels = shuffle_unison(
                                        preprocessed_data, preprocessed_labels)

        # Cross entropy loss treats each label as an index, so retransform the
        # labels to relative index given the number of classes
        preprocessed_labels = np.asarray(preprocessed_labels)
        self.label_encoder.fit(list(set(preprocessed_labels)))
        re_indexed_labels = self.label_encoder.transform(preprocessed_labels)

        return np.asarray(preprocessed_data), re_indexed_labels, preprocessed_labels, num_classes


    def get_train_task(self, num_classes):
        if num_classes == 0:
            num_classes = random.randint(2, self.max_num_classes)
        return self._get_task(self.train_loader, num_classes, selected_labels=[], num_samples=0)
        
    def get_test_task(self, selected_labels, num_samples):
        num_classes = len(selected_labels)
        return self._get_task(self.test_loader, num_classes, selected_labels=selected_labels, num_samples=num_samples)

class Reptile(object):
    def __init__(self, args):
        self.args = args
        self.model = LeNet()
        self.model.to(args.device)
        self.task_generator = TaskGen(args.max_num_classes)
        self.outer_stepsize = args.outer_stepsize
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.inner_stepsize)

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
           
            # self.model.zero_grad()
            self.optimizer.zero_grad()
            outputs = self.model(x[start:start+self.args.inner_batch_size])
            # print("output: {} - y: {}".format(outputs.shape, y.shape))
            loss = self.criterion(outputs, Variable(y[start:start+self.args.inner_batch_size].long()))
            total_loss += loss
            loss.backward()
            self.optimizer.step()
            # Similar to calling optimizer.step()
            # for param in self.model.parameters():
            #     param.data -= self.args.inner_stepsize * param.grad.data
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
            for iteration in range(self.args.n_iterations):
                # Generate task
                data, labels, original_labels, num_classes = self.task_generator.get_train_task(args.num_classes)
                
                weights_before = deepcopy(self.model.state_dict())
                
                loss = self.inner_training(data, labels, self.args.inner_iterations)
                total_loss += loss
                if iteration % self.args.log_every == 0:
                    print("-----------------------------")
                    print("iteration               {}".format(iteration+1))
                    print("Loss: {:.3f}".format(total_loss/(iteration+1)))
                    print("Current task info: ")
                    print("\t- Number of classes: {}".format(num_classes))
                    print("\t- Batch size: {}".format(len(data)))
                    print("\t- Labels: {}".format(set(original_labels)))
 
                    self.test()
                
                self._meta_gradient_update(iteration, num_classes, weights_before)
                    
        except KeyboardInterrupt:
            print("Manual Interrupt...")

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
        
        data, labels, _, _ = self.task_generator.get_test_task(selected_labels=[1,2,3,4,5], num_samples=100)
        predicted_labels = np.argmax(self.predict(data), axis=1)
        accuracy = np.mean(1*(predicted_labels==labels))*100

        print("Accuracy before few shots learning: {:.2f}%)\n----".format(accuracy))
        
        train_data, train_labels, _, _ = self.task_generator.get_test_task(selected_labels=[1,2,3,4,5], num_samples=1)
        
        weights_before = deepcopy(self.model.state_dict()) # save snapshot before evaluation
        for i in range(4):
            self.inner_training(train_data, train_labels, self.args.inner_iterations_test)

            data, labels, _, _ = self.task_generator.get_test_task(selected_labels=[1,2,3,4,5], num_samples=100)
            predicted_labels = np.argmax(self.predict(data), axis=1)
            accuracy = np.mean(1*(predicted_labels==labels))*100
            
            print("Accuracy after {} shot(s) learning: {:.2f}%)".format(
                        (i+1)*self.args.inner_iterations_test, accuracy))

        self.model.load_state_dict(weights_before) # restore from snapshot
            
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Meta Learning for MNIST')
    parser.add_argument('--inner-stepsize', type=float, default=1E-3,
                        help='stepsize in inner optimizer')
    parser.add_argument('--inner-iterations', type=int, default=5,
                        help='Number of iterations inner training')
    parser.add_argument('--inner-iterations-test', type=int, default=1,
                        help='Number of iterations inner training at test time')
    parser.add_argument('--inner-batch-size', type=int, default=5,
                        help='Inner Batch Size')
    parser.add_argument('--outer-stepsize', type=float, default=1E-1,
                        help='stepsize of outer optimization, i.e., meta-optimization')
    parser.add_argument('--n-iterations', type=int, default=100000,
                        help='number of outer updates; each iteration we sample one task and update on it')
    parser.add_argument('--max-num-classes', type=int, default=9,
                        help='Max number of classes in the training set')
    parser.add_argument('--device', type=str, default='cpu',
                        help='HW acceleration')
    parser.add_argument('--log-every', type=int, default=100,
                        help="Show progress every n iterations")
    # parser.add_argument('--num-test-classes', type=int, default=5,
    #                     help='Number of classes for test tasks')
    parser.add_argument('--num-classes', type=int, default=5,
                    help='Number of classes for training tasks')

    args = parser.parse_args()

    task_generator = TaskGen(args.max_num_classes)
    reptile = Reptile(args)
    reptile.meta_training()
    reptile.test()