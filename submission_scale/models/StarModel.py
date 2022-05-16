import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import unnormalize
import numpy as np


# Backbone/low level feature learner
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=(1, 1)), nn.BatchNorm2d(32), nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))

        self.conv2_1 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2_2 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2))

        self.conv3_1 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv3_2 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU())

        self.conv4_1 = nn.Sequential(nn.Conv2d(256, 16, 3, padding=(1, 1)), nn.BatchNorm2d(16), nn.ReLU())

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = torch.flatten(x,1)
        return x

class Localizer(nn.Module):
    def __init__(self):
        super(Localizer, self).__init__()
        self.fc1 = nn.Linear(16*12*12, 128)
        self.fc2 = nn.Linear(128,5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(16*12*12, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc2(x))
        return x

class StarModel(nn.Module):
    def __init__(self):
        super(StarModel, self).__init__()
        self.backbone = Backbone()
        self.localizer = Localizer()
        self.classifier = Classifier()
        self.head = 'localizer'

    def forward(self, x):
        x = self.backbone(x)
        if self.head == 'localizer':
            return self.localizer(x)
        else:
            return self.classifier(x)

    # Inference function
    def predict(self, x):
        self.head = 'classification'
        with torch.no_grad():
            pred = self.forward(x)[0][0].cpu().numpy()
            pred = int(pred > 0.5)
            if pred:
                self.head = 'localizer'
                pred = self.forward(x)
                pred = unnormalize(np.squeeze(pred).cpu().numpy())
            else:
                pred = np.full(5, np.nan)
        return pred
