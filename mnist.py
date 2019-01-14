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

from models import *
from sklearn import preprocessing

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
    def __init__(self):
        self._transform = transform=torchvision.transforms.Compose([
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
        ])

        self.sample_data_size = 1024
        self.max_samples = 10
        self.trainset = datasets.MNIST(root='./data',
                                        train=True,
                                        download=True,
                                        transform=self._transform)
        # self.sample_data_size = 2
        self.loader = torch.utils.data.DataLoader(self.trainset,
                                            batch_size=self.sample_data_size,
                                            shuffle=True)
        self.label_encoder = preprocessing.LabelEncoder()

    def get_task(self):
        num_classes = random.randint(2, TOT_CLASSES)
        # num_classes = 10
        selected_labels = random.sample(range(0, TOT_CLASSES), num_classes)
        _, (data, labels) = next(enumerate(self.loader))
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
        preprocessed_labels = self.label_encoder.transform(preprocessed_labels)

        return np.asarray(preprocessed_data), preprocessed_labels, num_classes

if __name__ == '__main__':
    task_generator = TaskGen()
    device = torch.device('cpu')
    model_f = GraySqueezeNetFeatures()

    # Keep different classifier models depending on the number of classes
    classifiers_dictionary = {}

    def train_on_batch(x, y, classifier):
        criterion = nn.CrossEntropyLoss()

        x = torch.tensor(x, dtype=torch.float, device=device)
        y = torch.tensor(y, dtype=torch.float, device=device)
        print(y)
        model_f.zero_grad()
        classifier.zero_grad()

        features = model_f(x)
        outputs = classifier(features)
        # print("output: {} - y: {}".format(outputs.shape, y.shape))
        loss = criterion(outputs, Variable(y.long()))
        loss.backward()
        # optimizer.step()
        for param in classifier.parameters():
            param.data -= innerstepsize * param.grad.data
        for param in model_f.parameters():
            param.data -= innerstepsize * param.grad.data

    def predict(x, f, c):
        x = torch.tensor(x, dtype=torch.float, device=device)
        # features = model_f(x)
        # outputs = model_c(features)
        # return outputs.data.numpy()
        features = f(x)
        outputs = c(features)
        return outputs.data.numpy()

    innerstepsize = 0.02 # stepsize in inner SGD
    innerepochs = 1 # number of epochs of each inner SGD
    outerstepsize0 = 0.1 # stepsize of outer optimization, i.e., meta-optimization
    niterations = 20 # number of outer updates; each iteration we sample one task and update on it

    # Reptile training loop
    for iteration in range(niterations):
        # Generate task
        data, labels, num_classes = task_generator.get_task()
        print("Num classes: {}".format(num_classes))

        if num_classes not in classifiers_dictionary.keys():
            print("yo")
            current_classifier = GraySqueezeNetClassifier(num_classes=num_classes)
            classifiers_dictionary[num_classes] = current_classifier
        else:
            current_classifier = classifiers_dictionary[num_classes]

        weights_f_before = deepcopy(model_f.state_dict())
        weights_c_before = deepcopy(current_classifier.state_dict())
        for _ in range(innerepochs):
            train_on_batch(data, labels, current_classifier)
        # Interpolate between current weights and trained weights from this task
        # I.e. (weights_before - weights_after) is the meta-gradient
        weights_f_after = model_f.state_dict()
        weights_c_after = current_classifier.state_dict()
        outerstepsize = outerstepsize0 * (1 - iteration / niterations) # linear schedule
        model_f.load_state_dict({name :
            weights_f_before[name] + (weights_f_after[name] - weights_f_before[name]) * outerstepsize
            for name in weights_f_before})
        current_classifier.load_state_dict({name :
            weights_c_before[name] + (weights_c_after[name] - weights_c_before[name]) * outerstepsize
            for name in weights_c_before})
        print("-----------------------------")
        print("iteration               {}".format(iteration+1))

    print(predict(data[0][None, :, :, :], model_f, current_classifier))
    print(labels)
    plt.imshow(data[0][0])
    plt.show()
