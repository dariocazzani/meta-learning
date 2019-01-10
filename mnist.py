import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
from torch import nn, autograd as ag
import torch.nn.init as init

import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy



seed = 0
random.seed(seed)
rng = np.random.RandomState(seed)
torch.manual_seed(seed)

TOT_CLASSES = 10

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class GraySqueezeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GraySqueezeNet, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)

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

    def get_task(self):
        num_classes = random.randint(2, TOT_CLASSES)
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

        return np.asarray(preprocessed_data), np.asarray(preprocessed_labels)

if __name__ == '__main__':
    task_generator = TaskGen()
    device = torch.device('cpu')
    model = GraySqueezeNet(num_classes=TOT_CLASSES)

    def train_on_batch(x, y):
        criterion = nn.CrossEntropyLoss()

        x = torch.tensor(x, dtype=torch.float, device=device)
        y = torch.tensor(y, dtype=torch.float, device=device)
        model.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y.long())
        loss.backward()
        # optimizer.step()
        for param in model.parameters():
            param.data -= innerstepsize * param.grad.data

    def predict(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        return model(x).data.numpy()

    plot = True
    innerstepsize = 0.02 # stepsize in inner SGD
    innerepochs = 1 # number of epochs of each inner SGD
    outerstepsize0 = 0.1 # stepsize of outer optimization, i.e., meta-optimization
    niterations = 20 # number of outer updates; each iteration we sample one task and update on it

    # Reptile training loop
    for iteration in range(niterations):
        weights_before = deepcopy(model.state_dict())
        # Generate task
        data, labels = task_generator.get_task()
        for _ in range(innerepochs):
            train_on_batch(data, labels)
        # Interpolate between current weights and trained weights from this task
        # I.e. (weights_before - weights_after) is the meta-gradient
        weights_after = model.state_dict()
        outerstepsize = outerstepsize0 * (1 - iteration / niterations) # linear schedule
        model.load_state_dict({name :
            weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize
            for name in weights_before})
        print("-----------------------------")
        print("iteration               {}".format(iteration+1))

    print(predict(data[0][None, :, :, :]))
    print(labels)
    plt.imshow(data[0][0])
    plt.show()
