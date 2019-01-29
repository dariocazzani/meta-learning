import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
from torch import nn, autograd as ag
from torch.autograd import Variable

import torch.nn.init as init

import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from models import LeNetFeatures, LeNetClassifier
from sklearn import preprocessing

import argparse
seed = 42
random.seed(seed)
rng = np.random.RandomState(seed)
torch.manual_seed(seed)

TOT_CLASSES = 10

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
        self.max_samples = 10
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

    def get_train_task(self, num_classes=0):
        if num_classes == 0:
            num_classes = random.randint(2, self.max_num_classes)
        # num_classes = 10
        selected_labels = random.sample(range(0, self.max_num_classes), num_classes)
        _, (data, labels) = next(enumerate(self.train_loader))
        data = data.numpy()
        labels = labels.numpy()

        preprocessed_labels = []
        preprocessed_data = []
        for c in range(num_classes):
            num_samples = random.randint(1, self.max_samples)
            avail_data_idx = np.where(labels==selected_labels[c])[0]
            random.shuffle(avail_data_idx)
            avail_data_idx = avail_data_idx[:num_samples]
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

    def get_test(self):
        _, (data, labels) = next(enumerate(self.train_loader))
        data = data.numpy()
        labels = labels.numpy()

        return data, labels


class Reptile(object):
    def __init__(self, args):
        self.args = args
        self.model_f = LeNetFeatures()
        self.model_f.to(args.device)
        self.task_generator = TaskGen(args.max_num_classes)
        self.outer_stepsize = args.outer_stepsize

        # Keep different classifier models depending on the number of classes
        self.classifiers_dictionary = {}
        self._init_classifiers()
        # tmp variable for classifier
        self.current_classifier = None

        self.criterion = nn.CrossEntropyLoss()

    def _init_classifiers(self):
        for num_classes in range(2, self.args.max_num_classes+1):
            classifier = LeNetClassifier(num_classes=num_classes)
            classifier.to(self.args.device)
            self.classifiers_dictionary[num_classes] = classifier

    def inner_training(self, x, y):
        """
        Run training on task
        """
        x, y = shuffle_unison(x, y)

        self.model_f.train()
        self.current_classifier.train()

        x = torch.tensor(x, dtype=torch.float, device=self.args.device)
        y = torch.tensor(y, dtype=torch.float, device=self.args.device)

        self.model_f.zero_grad()
        self.current_classifier.zero_grad()

        for start in range(0, len(x), self.args.inner_batch_size):
            features = self.model_f(x[start:start+self.args.inner_batch_size])
            outputs = self.current_classifier(features)
            # print("output: {} - y: {}".format(outputs.shape, y.shape))
            loss = self.criterion(outputs, Variable(y[start:start+self.args.inner_batch_size].long()))
            loss.backward()
            # Similar to calling optimizer.step()
            for param in self.current_classifier.parameters():
                param.data -= self.args.inner_stepsize * param.grad.data
            for param in self.model_f.parameters():
                param.data -= self.args.inner_stepsize * param.grad.data
        return loss

    def _meta_gradient_update(self, iteration, num_classes, weights_f_before, weights_c_before):
        """
        Interpolate between current weights and trained weights from this task
        I.e. (weights_before - weights_after) is the meta-gradient

            - iteration: current iteration - used for updating outer_stepsize
            - num_classes: current classifier number of classes
            - weights_f_before: state of weights of features net before inner steps training
            - weights_c_before: state of weights of classifier net before inner steps training
        """
        weights_f_after = self.model_f.state_dict()
        weights_c_after = self.current_classifier.state_dict()
        outer_stepsize = self.outer_stepsize * (1 - iteration / self.args.n_iterations) # linear schedule
        
        self.model_f.load_state_dict({name :
            weights_f_before[name] + (weights_f_after[name] - weights_f_before[name]) * outer_stepsize
            for name in weights_f_before})
        
        self.current_classifier.load_state_dict({name :
            weights_c_before[name] + (weights_c_after[name] - weights_c_before[name]) * outer_stepsize
            for name in weights_c_before})
        
        # Replace old classifier
        self.classifiers_dictionary[num_classes] = self.current_classifier


    def meta_training(self):
        # Reptile training loop
        total_loss = 0
        for iteration in range(self.args.n_iterations):
            # Generate task
            data, labels, original_labels, num_classes = self.task_generator.get_train_task()
            self.current_classifier = self.classifiers_dictionary[num_classes]

            weights_f_before = deepcopy(self.model_f.state_dict())
            weights_c_before = deepcopy(self.current_classifier.state_dict())

            for _ in range(self.args.inner_epochs):
                loss = self.inner_training(data, labels)
                total_loss += loss
            if iteration % self.args.log_every == 0:
                print("-----------------------------")
                print("iteration               {}".format(iteration+1))
                print("Loss: {:.3f}".format(total_loss/(iteration+1)))
                print("Current task info: ")
                print("\t- Number of classes: {}".format(num_classes))
                print("\t- Batch size: {}".format(len(data)))
                print("\t- Labels: {}".format(set(original_labels)))
            
            self._meta_gradient_update(iteration, num_classes, weights_f_before, weights_c_before)


            
           
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Meta Learning for MNIST')
    parser.add_argument('--inner-stepsize', type=float, default=1E-3,
                        help='stepsize in inner SGD')
    parser.add_argument('--inner-epochs', type=int, default=5,
                        help='umber of epochs of each inner SGD')
    parser.add_argument('--inner-batch-size', type=int, default=5,
                        help='Inner Batch Size')
    parser.add_argument('--outer-stepsize', type=float, default=1E-1,
                        help='stepsize of outer optimization, i.e., meta-optimization')
    parser.add_argument('--n-iterations', type=int, default=4000,
                        help='number of outer updates; each iteration we sample one task and update on it')
    parser.add_argument('--max-num-classes', type=int, default=10,
                        help='Max number of classes in the training set')
    parser.add_argument('--device', type=str, default='cpu',
                        help='HW acceleration')
    parser.add_argument('--log-every', type=int, default=20,
                        help="Show progress every n iterations")

    args = parser.parse_args()

    task_generator = TaskGen(args.max_num_classes)
    reptile = Reptile(args)
    reptile.meta_training()

    # model_f = LeNetFeatures()
    # model_f.to(device)

    # # Keep different classifier models depending on the number of classes
    # classifiers_dictionary = {}

    # def train_on_sampled_data(x, y, classifier):
    #     model_f.train()
    #     classifier.train()
    #     criterion = nn.CrossEntropyLoss()

    #     x = torch.tensor(x, dtype=torch.float, device=device)
    #     y = torch.tensor(y, dtype=torch.float, device=device)
    #     model_f.zero_grad()
    #     classifier.zero_grad()

    #     for start in range(0, len(x), inner_batch_size):
    #         features = model_f(x[start:start+inner_batch_size])
    #         outputs = classifier(features)
    #         # print("output: {} - y: {}".format(outputs.shape, y.shape))
    #         loss = criterion(outputs, Variable(y[start:start+inner_batch_size].long()))
    #         loss.backward()
    #         # optimizer.step()
    #         for param in classifier.parameters():
    #             param.data -= innerstepsize * param.grad.data
    #         for param in model_f.parameters():
    #             param.data -= innerstepsize * param.grad.data

    #     return loss

    # def predict(x, f, c):
    #     f.eval()
    #     c.eval()
    #     x = torch.tensor(x, dtype=torch.float, device=device)
    #     features = f(x)
    #     outputs = c(features)
    #     return outputs.cpu().data.numpy()

    # # Reptile training loop
    # total_loss = 0
    # for iteration in range(n_iterations):
    #     # Generate task
    #     data, labels, original_labels, num_classes = task_generator.get_train_task()

    #     if num_classes not in classifiers_dictionary.keys():
    #         current_classifier = LeNetClassifier(num_classes=num_classes)
    #         classifiers_dictionary[num_classes] = current_classifier.to(device)
    #     else:
    #         current_classifier = classifiers_dictionary[num_classes]

    #     weights_f_before = deepcopy(model_f.state_dict())
    #     weights_c_before = deepcopy(current_classifier.state_dict())
    #     for _ in range(innerepochs):
    #         loss = train_on_sampled_data(data, labels, current_classifier)
    #         total_loss += loss
    #     if iteration % 20 == 0:
    #         print("-----------------------------")
    #         print("iteration               {}".format(iteration+1))
    #         print("Loss: {:.3f}".format(total_loss/(iteration+1)))

    #     # Interpolate between current weights and trained weights from this task
    #     # I.e. (weights_before - weights_after) is the meta-gradient
    #     weights_f_after = model_f.state_dict()
    #     weights_c_after = current_classifier.state_dict()
    #     outerstepsize = outerstepsize0 * (1 - iteration / n_iterations) # linear schedule
    #     model_f.load_state_dict({name :
    #         weights_f_before[name] + (weights_f_after[name] - weights_f_before[name]) * outerstepsize
    #         for name in weights_f_before})
    #     current_classifier.load_state_dict({name :
    #         weights_c_before[name] + (weights_c_after[name] - weights_c_before[name]) * outerstepsize
    #         for name in weights_c_before})
    #     classifiers_dictionary[num_classes] = current_classifier


    # # Test
    # """
    #     1. Create task from test set.
    #     2. Reload feature extractor and classifier with the right number of classes
    #     3. Train for one iteration on the test set and predict
    # """
    # data, labels, original_labels, num_classes = task_generator.get_train_task(num_classes=10)
    # current_classifier = classifiers_dictionary[num_classes]
    # # train_on_sampled_data(data, labels, current_classifier)

    # data, labels = task_generator.get_test()
    # predicted_labels = np.argmax(predict(data, model_f, current_classifier),axis=1)
    # print(predicted_labels)
    # print(labels)
    # # print(original_labels)

    # print("Accuracy before few shot learning: {:2f}%)".format(np.mean(1*(predicted_labels==labels))*100))
    # plt.imshow(data[0][0])
    # plt.show()
