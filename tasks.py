import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

from helpers.datasets import KMNIST
from helpers.ops import shuffle_unison

from sklearn import preprocessing
import random
import numpy as np

class TaskGen(object):
    def __init__(self, args):
        self._transform = torchvision.transforms.Compose([
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
        ])
        self.use_kmnist = args.use_kmnist
        self.max_num_classes = args.max_num_classes
        self.max_samples_per_class = args.max_samples_per_class

        self.sample_data_size = self.max_num_classes * self.max_samples_per_class * 3
        
        # Combine kmnist and mnist in one dataset
        if self.use_kmnist:
            kmnist = KMNIST(root='./kmnist_data',
                                            train=True,
                                            download=True,
                                            transform=self._transform)

            mnist = datasets.MNIST(root='./mnist_data',
                                    train=True,
                                    download=True,
                                    transform=self._transform)

            # labels go from 0 to 9 for both datasets.
            # Add 10 to mnist labels so that we have 20 classes with labels from 0 to 19
            mnist.train_labels += 10 

            self.trainset = torch.utils.data.ConcatDataset([kmnist, mnist])
        else:
            self.trainset = datasets.MNIST(root='./mnist_data',
                                            train=True,
                                            download=True,
                                            transform=self._transform)

        self.testset = datasets.MNIST(root='./mnist_data',
                                        train=False,
                                        download=True,
                                        transform=self._transform)
        
        num_test = len(self.testset)
        indices = list(range(num_test))
        enroll_size = 0.2
        split = int(np.floor(enroll_size * num_test))

        np.random.seed(42)
        np.random.shuffle(indices)

        test_idx, enroll_idx = indices[split:], indices[:split]

        test_sampler = SubsetRandomSampler(test_idx)
        enroll_sampler = SubsetRandomSampler(enroll_idx)

        self.train_loader = torch.utils.data.DataLoader(self.trainset, 
                                    batch_size=self.sample_data_size, 
                                    shuffle=True)
        
        self.enroll_loader = torch.utils.data.DataLoader(self.testset, 
                                    batch_size=10000, 
                                    sampler=enroll_sampler)
                                    
        self.test_loader = torch.utils.data.DataLoader(self.testset,
                                    batch_size=10000,
                                    sampler=test_sampler)
        
        self.label_encoder = preprocessing.LabelEncoder()
    def _get_task(self, data_loader, num_classes, selected_labels, num_samples):
        """
            - data_loader
            - num_classes
            - selected_labels
            - num_samples
            - distribution: train, test, enroll
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
                class_num_samples = random.randint(2, self.max_samples_per_class)
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

    def get_enroll_task(self, selected_labels, num_samples):
        num_classes = len(selected_labels)
        return self._get_task(self.enroll_loader, num_classes, selected_labels=selected_labels, num_samples=num_samples)
        
    def get_test_task(self, selected_labels, num_samples):
        num_classes = len(selected_labels)
        return self._get_task(self.test_loader, num_classes, selected_labels=selected_labels, num_samples=num_samples)
