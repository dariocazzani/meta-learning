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
        re_indexed_labels = self.label_encoder.transform(preprocessed_labels)

        return np.asarray(preprocessed_data), re_indexed_labels, preprocessed_labels, num_classes

innerstepsize = 1E-3 # stepsize in inner SGD
innerepochs = 5 # number of epochs of each inner SGD
inner_batch_size = 5
outerstepsize0 = 0.1 # stepsize of outer optimization, i.e., meta-optimization
niterations = 20000 # number of outer updates; each iteration we sample one task and update on it


if __name__ == '__main__':
    task_generator = TaskGen()
    device = torch.device('cuda:2')
    model_f = LeNetFeatures()
    model_f.to(device)

    # Keep different classifier models depending on the number of classes
    classifiers_dictionary = {}

    def train_on_sampled_data(x, y, classifier):
        model_f.train()
        classifier.train()
        criterion = nn.CrossEntropyLoss()

        x = torch.tensor(x, dtype=torch.float, device=device)
        y = torch.tensor(y, dtype=torch.float, device=device)
        model_f.zero_grad()
        classifier.zero_grad()

        for start in range(0, len(x), inner_batch_size):
            features = model_f(x[start:start+inner_batch_size])
            outputs = classifier(features)
            # print("output: {} - y: {}".format(outputs.shape, y.shape))
            loss = criterion(outputs, Variable(y[start:start+inner_batch_size].long()))
            loss.backward()
            # optimizer.step()
            for param in classifier.parameters():
                param.data -= innerstepsize * param.grad.data
            for param in model_f.parameters():
                param.data -= innerstepsize * param.grad.data

        return loss

    def predict(x, f, c):
        f.eval()
        c.eval()
        x = torch.tensor(x, dtype=torch.float, device=device)
        features = f(x)
        outputs = c(features)
        return outputs.cpu().data.numpy()

    # Reptile training loop
    total_loss = 0
    for iteration in range(niterations):
        # Generate task
        data, labels, original_labels, num_classes = task_generator.get_task()

        if num_classes not in classifiers_dictionary.keys():
            current_classifier = LeNetClassifier(num_classes=num_classes)
            classifiers_dictionary[num_classes] = current_classifier.to(device)
        else:
            current_classifier = classifiers_dictionary[num_classes]

        weights_f_before = deepcopy(model_f.state_dict())
        weights_c_before = deepcopy(current_classifier.state_dict())
        for _ in range(innerepochs):
            loss = train_on_sampled_data(data, labels, current_classifier)
            total_loss += loss
        if iteration % 20 == 0:
            print("-----------------------------")
            print("iteration               {}".format(iteration+1))
            print("Loss: {:.3f}".format(total_loss/(iteration+1)))

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


    # Test
    """
        1. Create task from test set.
        2. Reload feature extractor and classifier with the right number of classes
        3. Train for one iteration on the test set and predict
    """
    data, labels, original_labels, num_classes = task_generator.get_task()
    current_classifier = classifiers_dictionary[num_classes]
    train_on_sampled_data(data, labels, current_classifier)
    print(predict(data[0][None, :, :, :], model_f, current_classifier))
    print(original_labels)
    print(labels)
    plt.imshow(data[0][0])
    plt.show()
