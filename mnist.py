import torch
import torchvision
import torchvision.datasets as datasets
import random
import numpy as np
import matplotlib.pyplot as plt


seed = 0
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

    def get_task(self):
        num_classes = random.randint(2, TOT_CLASSES+1)
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

task_generator = TaskGen()
data, labels = task_generator.get_task()
print(labels)
plt.imshow(data[1][0])
plt.show()
