import typing as t

from sklearn.utils import shuffle
from torchsummary import summary
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from utils import DEVICE, synthesize_data, normalize, unnormalize, unnormalize_tensor, to_corners, calc_iou_array
from losses import modulated_loss

# Backbone/low level feature learner
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=(1, 1)), nn.BatchNorm2d(32), nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))

        self.conv2_1 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2_2 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2))

        self.conv3_1 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv3_2 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU())

        self.conv4_1 = nn.Sequential(nn.Conv2d(512, 16, 3, padding=(1, 1)), nn.BatchNorm2d(16), nn.ReLU())

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

# Detects/regresses bounding box and angle
class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.fc1 = nn.Linear(16*12*12, 128)
        self.fc2 = nn.Linear(128,5)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

# Classifies if star or not
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

# Combine the 3 models
class StarModel(nn.Module):
    def __init__(self):
        super(StarModel, self).__init__()
        self.backbone = Backbone()
        self.detector = Detector()
        self.classifier = Classifier()
        self.head = 'detector'

    def forward(self, x):
        x = self.backbone(x)
        if self.head == 'detector':
            return self.detector(x)
        else:
            return self.classifier(x)

# Dataloader
# Normalizes the label
class StarDataset(torch.utils.data.Dataset):
    """Return star image and labels"""

    def __init__(self, data_size=50000, has_star=True):
        self.data_size = data_size
        self.has_star = has_star

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image, label = synthesize_data(has_star=self.has_star)
        label = normalize(label)
        return image[None], label


def train(model: StarModel, dl: StarDataset, num_epochs: int, optimizer, loss_fn, scheduler, localizer=True) -> StarModel:
    #Initiate model for training (Batchnorm and dropout)
    model.train()
    # Get model summary
    summary(model, (1,200,200))
    # Initialize total loss and iou

    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch}")
        epoch_losses = []
        epoch_ious = []
        for image, labels in tqdm(dl, total=len(dl)):
            image = image.to(DEVICE).float()
            labels = labels.to(DEVICE).float()
            optimizer.zero_grad()
            preds = model(image)
            if localizer:
                loss = loss_fn(
                    unnormalize_tensor(preds).to(DEVICE),
                    unnormalize_tensor(labels).to(DEVICE)
                )
                loss = torch.mean(loss)
                preds8 = to_corners(torch.unsqueeze(unnormalize_tensor(preds), 1).to(DEVICE))
                labels8 = to_corners(torch.unsqueeze(unnormalize_tensor(labels), 1).to(DEVICE))
                iou = calc_iou_array(preds8, labels8).to(DEVICE)
                epoch_ious.append(iou.cpu().detach().numpy())
                epoch_losses.append(loss.cpu().detach().numpy())
            else:
                loss = loss_fn(preds, labels)
                epoch_losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(epoch)
        print(np.mean(epoch_losses))
        if localizer:
            print(np.mean(epoch_ious))

    return model


def main():
    # Initialize model and set to available device
    model = StarModel().to(DEVICE)
    # Initialize loss functions and optimizer for localization
    optimizer = torch.optim.Adam(model.parameters())
    loss_localizer = modulated_loss
    epochs = 50
    batch_size = 128
    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    # Train the localizer
    star_model = train(
        model,
        torch.utils.data.DataLoader(StarDataset(data_size=64000), batch_size=batch_size, shuffle=True),
        num_epochs=epochs,
        optimizer=optimizer,
        loss_fn=loss_localizer, 
        scheduler=scheduler
    )
    # Freeze the backbone layers since they've been trained well enough on low level features during localizer training
    print("Classification Training")
    # Change head to classifier
    star_model.head = 'classifier'
    loss_classifier = torch.nn.BCELoss()

    for param in star_model.backbone.parameters():
        param.requires_grad = False

    for param in star_model.localizer.parameters():
        param.requires_grad = False

    epochs=5
    optimizer = torch.optim.Adam(star_model.parameters())
    star_model = train(
        star_model,
        torch.utils.data.DataLoader(StarDataset(data_size=32000, has_star=None), batch_size=batch_size, shuffle=True),
        num_epochs=epochs,
        optimizer=optimizer,
        loss_fn=loss_classifier, 
        scheduler=scheduler
    )
    torch.save(star_model.state_dict(), "model.pickle")




if __name__ == "__main__":
    main()
