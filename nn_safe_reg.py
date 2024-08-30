import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
class NNDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        #read labels from CSV
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        #load depth array from image path
        image = torch.from_numpy(np.load(img_path))
        image = torch.unsqueeze(image,0)

        #state vector
        state = np.zeros(3)
        for i in range(3):
            state[i] = self.img_labels.iloc[idx, i + 2]

        #safe force limit - ground truth labels
        label = np.zeros(18)
        for i in range(len(label)):
            label[i] = self.img_labels.iloc[idx, i + 6]

        #applt applicable transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, state, label

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        # conv layer 1
        self.conv11 = nn.Conv2d(1, 4, kernel_size=3)
        self.conv12 = nn.Conv2d(4, 4, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv layer 2
        self.conv21 = nn.Conv2d(4, 8, kernel_size=3)
        self.conv22 = nn.Conv2d(8, 8, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #linear layers (after concatenation of state vector)
        self.fc1 = nn.Linear(10955, 500)
        self.fc2 = nn.Linear(500, 150)
        self.fc3 = nn.Linear(150, 100)

        #final layer - 18 outputs for 18 vertices of the safe control reiong
        self.fc4 = nn.Linear(100, 18)

    def forward(self, image, state):
        #pass depth image through convolutional layers
        image = F.relu(self.conv11(image))
        image = F.relu(self.conv12(image))
        image = self.pool1(image)

        image = F.relu(self.conv21(image))
        image = F.relu(self.conv22(image))
        image = self.pool2(image)

        image = torch.flatten(image, 1)  # flatten all dimensions except the batch dimension

        #concatenate the state vecotr to the image vector
        state = state.float()
        combined = torch.cat((image.view(image.size(0), -1),
                              state.view(state.size(0), -1)), dim=1)

        #pass concatenated vector through linear layers
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        combined = F.relu(self.fc3(combined))

        #final network output
        combined = self.fc4(combined)

        return combined

